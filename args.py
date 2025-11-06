import argparse

def make_args():
    parser = argparse.ArgumentParser(description='SR-HAN main.py')
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--batch', type=int, default = 4096, metavar='N', help='input batch size for training')  #*
    parser.add_argument('--testB', type=int, default = 512, metavar='N', help='input batch size for testing')
    parser.add_argument('--seed', type=int, default = 29, metavar='int', help='random seed')
    parser.add_argument('--decay', type=float, default = 0.97, metavar='LR_decay', help='decay')
    parser.add_argument('--lr', type=float, default = 0.025, metavar='LR', help='learning rate')#*
    parser.add_argument('--minlr', type=float,default = 0.0001)
    parser.add_argument('--reg', type=float, default = 0.004)
    parser.add_argument('--epochs', type=int, default = 400, metavar='N', help='number of epochs to train')
    parser.add_argument('--patience', type=int, default = 5, metavar='int', help='early stop patience')
    parser.add_argument('--topk', type=int, default= 20)
    parser.add_argument('--hide_dim', type=int, default = 32, metavar='N', help='embedding size')#*
    parser.add_argument('--layer_dim',nargs='?', default ='[32]', help='Output size of every layer') #*
    parser.add_argument('--uiLayers', type =int, default = 2, help='the numbers of ui-GCN layer') #*
    parser.add_argument('--au_uiLayers', type = int, default = 2, help='the numbers of au_ui-GCN layer') #*
 
    #Diffusion model
    parser.add_argument('--mean_type', type=str, default='eps', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--mean_typeNon', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default = 8, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001)
    parser.add_argument('--noise_max', type=float, default=0.01)
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default = 5, help='steps for sampling/denoising')#10
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
    # params for the MLP
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--mlp_dims', type=str, default='[8]', help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default = False, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default = 10, help='timestep embedding size')
    parser.add_argument('--mlp_act_func', type=str, default = 'tanh', help='the activation function for MLP')
    parser.add_argument('--optimizer2', type=str, default = 'AdamW', help='optimizer for MLP: Adam, AdamW, SGD, Adagrad, Momentum')
    # params for the Autoencoder
    parser.add_argument('--n_cate', type=int, default = 3, help='category num of items')
    parser.add_argument('--in_dims', type=str, default = '[8]', help='the dims for the encoder')
    parser.add_argument('--out_dims', type=str, default = '[8]', help='the hidden dims for the decoder')
    parser.add_argument('--act_func', type=str, default = 'tanh', help='activation function for autoencoder')
    parser.add_argument('--lamda', type=float, default = 0.03, help='hyper-parameter of multinomial log-likelihood for AE: 0.01, 0.02, 0.03, 0.05')
    parser.add_argument('--optimizer1', type=str, default = 'AdamW', help='optimizer for AE: Adam, AdamW, SGD, Adagrad, Momentum')
    parser.add_argument('--anneal_cap', type=float, default = 0.005)
    parser.add_argument('--anneal_steps', type=int, default = 500)
    parser.add_argument('--vae_anneal_cap', type=float, default = 0.3)
    parser.add_argument('--vae_anneal_steps', type=int, default = 200)
    parser.add_argument('--reparam', type=bool, default = True, help="Autoencoder with variational inference or not")

   

    # loss
    parser.add_argument('--elbo_w', type=float, default = 0.035)
    parser.add_argument('--di_pre_w', type=float, default = 0.4)
    parser.add_argument('--con_fe_w', type=float, default = 0.02)
    parser.add_argument('--ssl_reg', type=float, default = 0.06)
    parser.add_argument('--ssl_temp', type=float, default = 0.65, help='the temperature in softmax')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    args = parser.parse_args()

    return args
