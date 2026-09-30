#new change
"""CPU unit tests for cohorts, independent draws, direction and magnitude."""
import unittest
import torch
from stand_force_recovery_sampling import curriculum_cohort, rear_draw, transform_peaks


class CurriculumTests(unittest.TestCase):
    def test_proportions_and_replay(self):
        ids=torch.arange(4096);k=torch.arange(4096)%13
        torch.manual_seed(6);p=torch.randn(4096,3);p=p/p.norm(dim=-1,keepdim=True)*3
        stiffness=torch.full((4096,1),200.)
        q,rear=transform_peaks(p,stiffness,.1,ids,k,42,'mixed');g=curriculum_cohort(ids,'mixed')
        self.assertEqual(torch.bincount(g).tolist(),[1024,2048,1024])
        self.assertEqual(torch.count_nonzero(q[g==0]),0)
        self.assertTrue(torch.equal(q[g==1],p[g==1]))
        torch.testing.assert_close(q[g==2].norm(dim=-1),torch.full((1024,),5.5))
        self.assertTrue(.45 < float(rear[g==2].float().mean()) < .55)
        self.assertTrue(torch.all(q[rear,0]<0))
        self.assertTrue(torch.all(torch.abs(torch.atan2(q[rear,1],-q[rear,0]))<=torch.pi/6+1e-6))
        torch.testing.assert_close(q[rear,2]/5.5,p[rear,2]/3)
        # Changing another environment's event index cannot change this one.
        changed=k.clone();changed[0]+=20
        r,b=transform_peaks(p,stiffness,.1,ids,changed,42,'mixed')
        self.assertTrue(torch.equal(q[1:],r[1:]));self.assertTrue(torch.equal(rear[1:],b[1:]))

    def test_eval_has_no_bias_and_retains_cap(self):
        p=torch.tensor([[2.,0,0],[0,4.,0]]);ids=torch.arange(2);k=torch.zeros(2,dtype=torch.long)
        for mode in ('low','medium'):
            q,rear=transform_peaks(p,torch.full((2,1),100.),.1,ids,k,7,mode)
            self.assertFalse(rear.any())
            torch.testing.assert_close(q,p if mode=='low' else torch.tensor([[4.,0,0],[0,7.,0]]))
        q,_=transform_peaks(p,torch.full((2,1),50.),.1,ids,k,7,'medium')
        torch.testing.assert_close(q.norm(dim=-1),torch.tensor([4.,5.]))
        self.assertNotEqual(rear_draw(42,3,0),rear_draw(42,3,1))


if __name__=='__main__':unittest.main()
