#new change
"""Full 540 trunk, zero context injection, new upper outputs; anchored PPO.

PPO loss/GAE conventions follow the installed RSL-RL implementation. This
independent class supports only fixed-LR, feedforward, single-GPU PPO.
"""
import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.algorithms import PPO
from stand_force_joint_v1_cfg import CHECKPOINT540,LEG_IDS,UPPER_IDS,ARM,WAIST

class WarmBodyNet(nn.Module):
    def __init__(self,context_dim,critic=False):
        super().__init__()
        self.lower=MLP(109,1 if critic else 12,[256,256,128],'elu')
        state=torch.load(CHECKPOINT540,map_location='cpu',weights_only=False)
        source=state['critic_state_dict' if critic else 'actor_state_dict']
        self.lower.load_state_dict({k[4:]:v for k,v in source.items() if k.startswith('mlp.')},strict=True)
        self.context=nn.Linear(context_dim,256,bias=False);nn.init.zeros_(self.context.weight)
        self.critic=critic
        if not critic:
            self.upper=nn.Linear(128,len(UPPER_IDS));nn.init.zeros_(self.upper.weight);nn.init.zeros_(self.upper.bias)
    def forward(self,x):
        h=self.lower[0](x[:,:109])+self.context(x[:,109:])
        for index in range(1,len(self.lower)-1):h=self.lower[index](h)
        low=self.lower[-1](h)
        if self.critic:return low
        out=low.new_zeros((len(low),29));out[:,LEG_IDS]=low;out[:,UPPER_IDS]=self.upper(h)
        return out

class JointModel(MLPModel):
    def __init__(self,obs,obs_groups,obs_set,output_dim,**kwargs):
        super().__init__(obs,obs_groups,obs_set,output_dim,**kwargs)
        assert self.obs_groups==['lower_policy','policy'] and not self.obs_normalization
        self.mlp=WarmBodyNet(self.obs_dim-109,critic=output_dim==1)
        if self.distribution is not None:
            # Fixed group-specific exploration; no accidental std escalation.
            assert self.distribution.std_type=='log'
            with torch.no_grad():
                self.distribution.log_std_param.fill_(torch.log(torch.tensor(.05)))
                self.distribution.log_std_param[LEG_IDS]=torch.log(torch.tensor(.02))
            self.distribution.log_std_param.requires_grad_(False)

class AnchoredPPO(PPO):
    def __init__(self,*args,anchor_coef=1.,**kwargs):
        super().__init__(*args,**kwargs)
        assert not self.actor.is_recurrent and not self.is_multi_gpu and self.rnd is None and self.symmetry is None
        assert self.schedule=='fixed' and not self.normalize_advantage_per_mini_batch
        self.anchor_coef=anchor_coef;self.teacher=None
        self.optimizer=torch.optim.Adam([p for m in (self.actor,self.critic) for p in m.parameters() if p.requires_grad],lr=self.learning_rate)
    def anchor_loss(self,obs,mean):
        assert self.teacher is not None
        with torch.no_grad():old=self.teacher(obs['lower_policy'])
        mask=obs['anchor_mask'][:,0]
        leg=((.30*(mean[:,LEG_IDS]-old)/.03).square()).mean(-1)
        upper=torch.tanh(mean[:,UPPER_IDS]).square().mean(-1)
        # Strong zero-force preservation, weaker anchor during force/recovery.
        weight=.05+.95*mask
        return (weight*(leg+upper)).mean()
    def update(self):
        totals={k:0. for k in ['value','surrogate','entropy','anchor','kl','clip_fraction']};count=0;early_stop=False
        for batch in self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs):
            self.actor(batch.observations,stochastic_output=True)
            logprob=self.actor.get_output_log_prob(batch.actions)
            values=self.critic(batch.observations)
            entropy=self.actor.output_entropy.mean()
            kl=self.actor.get_kl_divergence(batch.old_distribution_params,self.actor.output_distribution_params).mean()
            # A trust-region stop, not a LR change. Never mask a large update.
            if count and float(kl.detach())>.015:
                early_stop=True;break
            ratio=torch.exp(logprob-batch.old_actions_log_prob.squeeze(-1))
            adv=batch.advantages.squeeze(-1)
            surrogate=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
            if self.use_clipped_value_loss:
                clipped=batch.values+(values-batch.values).clamp(-self.clip_param,self.clip_param)
                value=torch.maximum((values-batch.returns).square(),(clipped-batch.returns).square()).mean()
            else:value=(values-batch.returns).square().mean()
            anchor=self.anchor_loss(batch.observations,self.actor.output_mean)
            loss=surrogate+self.value_loss_coef*value-self.entropy_coef*entropy+self.anchor_coef*anchor
            self.optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
            self.optimizer.step()
            stats={'value':value,'surrogate':surrogate,'entropy':entropy,'anchor':anchor,'kl':kl,
                   'clip_fraction':((ratio-1).abs()>self.clip_param).float().mean()}
            for key,v in stats.items():totals[key]+=float(v.detach())
            count+=1
        self.storage.clear()
        return {**{k:v/max(count,1) for k,v in totals.items()},'updates':float(count),'kl_stop':float(early_stop)}

def runner_config(seed):
    model=dict(class_name='stand_force_joint_v1_model:JointModel',hidden_dims=[256,256,128],activation='elu',obs_normalization=False)
    return dict(seed=seed,num_steps_per_env=48,save_interval=10,logger='tensorboard',
      obs_groups={'actor':['lower_policy','policy'],'critic':['lower_policy','policy']},
      actor={**model,'distribution_cfg':{'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.05,'std_type':'log'}},
      critic=dict(model),algorithm=dict(class_name='stand_force_joint_v1_model:AnchoredPPO',learning_rate=1e-5,schedule='fixed',
        num_learning_epochs=2,num_mini_batches=4,clip_param=.05,entropy_coef=0.,desired_kl=.005,gamma=.99,lam=.95,
        value_loss_coef=1.,max_grad_norm=1.,use_clipped_value_loss=True,rnd_cfg=None,symmetry_cfg=None,anchor_coef=1.))
