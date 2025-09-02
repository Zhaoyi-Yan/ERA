import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pdist

__all__ = ['L1Triplet', 'L2Triplet', 'ContrastiveLoss', 'RkdDistance', 'RKdAngle', 'DeepWalkEmbeddingLoss', 'HardDarkRank']


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p==2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed, negative_embed, p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student, teacher):
        if student.dim() == 2:
            student = student.unsqueeze(2).unsqueeze(3)
            teacher = teacher.unsqueeze(2).unsqueeze(3)

        return (self.transform(student) - teacher).pow(2).mean()


# normalized_x and normalized_y, the C dimension are reduced. B*(H*W), normalized the (H*W) dimension
class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean() # filter out the diag, ie, i=j
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

def deepwalk_filter(evals, window):
    evals = torch.where(evals >= 1, torch.tensor(1.0), evals * (1 - evals.pow(window)) / (1 - evals) / window)
    evals = torch.clamp(evals, min=0)
#     logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f", torch.max(evals), torch.min(evals))
    return evals


def approximate_normalized_graph_laplacian(A):
    n = A.shape[0]
    d = A.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
    
    # Convert to double precision
    A = A.to(torch.float64)
    D_inv_sqrt = D_inv_sqrt.to(torch.float64)
    
    # Compute the normalized Laplacian
    L = torch.eye(n).to(A) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
    
    # Add regularization
    L += 1e-3 * torch.eye(n).to(A)
    is_success = 0
    try:
        evals, evecs = torch.linalg.eigh(L)
        is_success=1
    except:
        raise RuntimeError("Eigen decomposition did not converge")

    D_rt_invU = D_inv_sqrt @ evecs
    return evals, D_rt_invU


# def approximate_normalized_graph_laplacian(A):
#     n = A.shape[0]
#     d = A.sum(dim=1)
#     D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
    
    
    
#     L = torch.eye(n).to(A) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
#     evals, evecs = torch.linalg.eigh(L + 1e-4*torch.eye(n).to(A))
# #     evals, evecs = torch.linalg.eigh(L)

#     D_rt_invU = D_inv_sqrt @ evecs
#     return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = torch.mm(torch.diag(torch.sqrt(evals)), D_rt_invU.T).T
    mmT = torch.mm(X, X.T) * (vol / b)
    Y = torch.log(torch.clamp(mmT, min=1))
#     logger.info("Computed DeepWalk matrix with %d non-zero elements", torch.count_nonzero(Y))
    return Y

def svd_deepwalk_matrix(X, dim):
    u, s, _ = torch.svd(X)
    return torch.mm(torch.diag(torch.sqrt(s[:dim])), u[:, :dim].T).T

class DeepWalkEmbeddingLoss(nn.Module):
    def __init__(self, window, dim, negative):
        super().__init__()
        self.window = window
        self.dim = dim
        self.negative = negative
        self.skip_count = 0

    def forward(self, stu, tec):
        """
        X: B*C;
        """
        # build adjacency matrix for tec
        with torch.no_grad():
            t_d = pdist(tec, squared=False)
            mean_td = t_d[t_d>0].mean() # filter out the diag, ie, i=j
            t_d = t_d / mean_td
        # build adjacency matrix for stu
        d = pdist(stu, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d
        
        try:
            with torch.no_grad():
                t_vol = float(t_d.sum())
                # Perform eigen-decomposition of D^{-1/2} A D^{-1/2}, keep top rank eigenpairs
                tec_evals, tec_D_rt_invU = approximate_normalized_graph_laplacian(t_d)
                # Approximate deepwalk matrix
                tec_deepwalk_matrix = approximate_deepwalk_matrix(tec_evals, tec_D_rt_invU, window=self.window, vol=t_vol, b=self.negative)
                # Factorize deepwalk matrix with SVD
                tec_deepwalk_embedding = svd_deepwalk_matrix(tec_deepwalk_matrix, dim=self.dim)


            s_vol = float(d.sum())
            # Perform eigen-decomposition of D^{-1/2} A D^{-1/2}, keep top rank eigenpairs
            stu_evals, stu_D_rt_invU = approximate_normalized_graph_laplacian(d)
            # Approximate deepwalk matrix
            stu_deepwalk_matrix = approximate_deepwalk_matrix(stu_evals, stu_D_rt_invU, window=self.window, vol=s_vol, b=self.negative)
            # Factorize deepwalk matrix with SVD
            stu_deepwalk_embedding = svd_deepwalk_matrix(stu_deepwalk_matrix, dim=self.dim)

            loss = F.smooth_l1_loss(stu_deepwalk_embedding, tec_deepwalk_embedding, reduction='mean')
        except Exception as e:
            self.skip_count += 1
            print(f'skipping this batch deepwalk loss!, current skip count: {self.skip_count}')
            return torch.nn.Parameter(torch.tensor([0.0])).to(stu)
            
        
        return loss