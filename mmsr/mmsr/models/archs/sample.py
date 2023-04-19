def forward(self, Q, K, V, mask):
        l = Q.shape[-2]
        if self.multi_gauss:
            assert not self.key2
            attn = self.multi_gauss_kernel(Q, K, mask)
        else:
            # if self.norm:
            #     Q = Q/torch.norm(Q, p = 2, keepdim= True, dim = -1)
            #     K = K/torch.norm(K, p = 2, keepdim= True, dim = -1)
            if not self.key2:
                dot = torch.matmul(Q, torch.transpose(K, -2, -1))
                dot = dot / math.sqrt(self.head_dim)
                dot = dot - 1e6 * (1 - mask[:, None, None, :])
                attn = nn.functional.softmax(dot, dim = -1)
                # pdb.set_trace()
            else:
                
                K1 = K[:, :, :, : self.head_dim]
                K2 = K[:, :, :, self.head_dim :]
 
                #### check every step of this, consider numerical issue of dist_min = torch.tensor(1e6)
                if self.hard_em:
                    QK1_distance = self.dist._sq_dist(Q, K1, postprocess = False)
                    QK2_distance = self.dist._sq_dist(Q, K2, postprocess = False)
                    dist_min = torch.minimum(QK1_distance, QK2_distance)
                    attn = nn.functional.softmax((-1/(2*self.var))*dist_min - 1e6 * (1 - mask[:, None, None, :]), dim = -1)
                    # pdb.set_trace()
                ### check every step of this, consider numerical issue
                elif self.soft_em:
                    max_l = (mask.sum(0)!=0.).sum()
                    pi = self.pi.clone().detach()
                    QK1_distance = (-1/(2*self.var))*self.dist._sq_dist(Q, K1, postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                    QK2_distance = (-1/(2*self.var))*self.dist._sq_dist(Q, K2, postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                
 
                    attn = torch.exp(QK1_distance)*pi[None, :, None,:l] + (1 - pi[None, :, None, :l])*torch.exp(QK2_distance)
                    
                    
                    if self.training:
                    #update self.pi, using mask
                        
                        N1 = torch.einsum('nl,nhlk->hk',mask,(torch.exp(QK1_distance)*pi[None, :, None, :l])/(attn + 1e-6))
                        # N2 = torch.einsum('nhlk->hk',torch.exp(QK2_distance)*pi[None, :, :l, None]/(attn + 1e-6))
                        N = torch.einsum('ln,nk->k', mask.T, mask)[None, :] + 1e-6
                        
                        #(h,l)
                        pi_new = self.pi.clone().detach()
                        pi_new[:, :max_l] = (N1/N).detach()[:,:max_l]
                        pi_new.to(Q)
                        # print(N1, 'hihi')
 
                        self.pi.copy_(pi_new.detach())
 
                    
                    attn = attn/(attn.sum(dim = -1)[:, :, :, None])
                    if self.track_kk:
                        kk_distance = self.dist._sq_dist(K1, K2, postprocess = False).detach().mean()
                        kk_distance.to(Q)
                        self.kk_distance.copy_(kk_distance)
                    # pdb.set_trace()
 
                else: 
                    ###find a suitable way to make self.pi learnable here
                    QK1_distance = (-1/(2*self.var))*self.dist._sq_dist(Q, K1, postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                    QK2_distance = (-1/(2*self.var))*self.dist._sq_dist(Q, K2, postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                    
#                     attn = torch.exp(QK1_distance)*torch.clamp(self.pi, min = 0., max = 1.)[0][None, :, None, :l] + torch.exp(QK2_distance)*torch.clamp(self.pi, min = 0., max = 1.)[1][None, :, None, :l]
                    attn = torch.exp(QK1_distance)*torch.clamp(self.pi, min = 0., max = 1.)[0] + torch.exp(QK2_distance)*torch.clamp(self.pi, min = 0., max = 1.)[1]
                    attn = attn/(attn.sum(dim = -1)[:, :, :, None])
 
                    if self.track_kk:
                        kk_distance = self.dist._sq_dist(K1, K2, postprocess = False).detach().mean()
                        kk_distance.to(Q)
#                         self.kk_distance.copy_(kk_distance)
                  
                    # pdb.set_trace()
#         self.kk_distance = kk_distance.mean()
#         self.attn_matrix = attn
        X = torch.matmul(self.drop_attn(attn), V)
        return X