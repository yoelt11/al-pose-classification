import torch
from torch import nn
from .LinearProjection import LinearProjection

"""
Code based on: Action Transformer: A Self-Attention Model for Short-Time Pose-Based
Human Action Recognition
From Author: Vittori Mazzia, Simone Angarano1, Francesco Salvetti, Federico Angelin and Marcello Chiaberge
"""
class AcT(nn.Module):
	""" This layer creates a linear projection, it also appends the tokens and adds the position embeddings. 
		This is the previous step before the encoder: [x_bcls, X_2Dpose] + X_pos  
		
		Parameters:
		-----------
			- B: batch
			- T: Frame number
			- C: Channel number
			- N: Node number
			- n_head: the number of heads in the multiattention layer of the encoder
			- n_layers: the number of layers in the encoder
			- d_last_mlp: the output dimension of the last multilayer perceptron layer
			- classes = the number of classes
		Attributes
		----------:
			- d_model: as per paper this dimension depends on the nhead
			- d_mlp: as per paper this dimension depends on d_model
			- X_pos = position embedding
			"""
	def __init__(self, B=40, T=5, N=17, C=3, nhead=1, num_layer=4, d_last_mlp=256, classes=20, device='cpu'):

		super().__init__()
		#print(f"[Info] pose classification is in training mode: {self.training} running on device: {device}")	

		# Device
		self.device = torch.device(device)
		
		# Attributes
		self.d_model = nhead * 64  # as per paper
		self.d_mlp = self.d_model * 4  # as per paper
		self.B = B
		self.num_heads = nhead
		
		dropout = 0.0 # does not have a lot of impact in network

		# Linear projection
		self.linear_projection = LinearProjection(B,T,N,C,self.d_model, device)

		# Encoder
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_mlp, nhead=nhead, dropout=dropout, norm_first=False, batch_first=True, activation="gelu")
		self.layer_norm = nn.LayerNorm([T + 1, self.d_model],eps=1e-6) # not sure if needed
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer, norm=self.layer_norm)

		# Multi-layer perceptron
		if B == 1:
			self.mlp = nn.Sequential(
						nn.Linear(self.d_model, d_last_mlp),
						nn.GELU(),
						nn.Dropout(dropout),
						nn.Linear(d_last_mlp, classes),
						nn.LogSoftmax(dim=0)
						)
		else: 
			self.mlp = nn.Sequential(
						nn.Linear(self.d_model, d_last_mlp),
						nn.GELU(),
						nn.Dropout(dropout),
						nn.Linear(d_last_mlp, classes),
						nn.LogSoftmax(dim=1)
						)
			
	def info(self):
		print(f"[INFO][PoseClassification] is in training mode: {self.training} running on device: {self.device}")	
	
	def generate_mask(self, X_in):
		# Experimental random masking for the input of the encoder
		B, T, D, C= X_in.shape
		mask = torch.FloatTensor(T + 1, T + 1).uniform_() > 0.8
		return mask.to(self.device)

	def forward(self, X_in):
		# -- linear projection	
		lp = self.linear_projection(X_in) # [B, T+1, D_model]
		# -- apply mask if training
		if self.training:
			mask = self.generate_mask(X_in.clone().detach())
			enc_out = self.transformer_encoder(lp, mask=mask) # [B, T+1, D_model]
		else:
			enc_out = self.transformer_encoder(lp) # [B, T+1, D_model]
		out =  self.mlp(enc_out[:, 0, :].squeeze())

		return out

"""
---------------------------------------	
	Testing Zone
---------------------------------------	
"""
if __name__=="__main__":
	
	B, T, N, C = 40, 30, 17, 3
	X_in = torch.randn([B,T,N,C])
	print(X_in.shape)
	model = AcT(B,T,N,C,1,20)
	
	out = model(X_in)

	print(out)
