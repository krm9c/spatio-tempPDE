from torch.utils.tensorboard import SummaryWriter
import copy
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np_
import pickle

class Trainer(eqx.Module):
    writer: SummaryWriter
    loss: str
    problem:str
    metric: str
    dict: dict()

    def __init__(self, logdir="runs", Loss='class', metric='class', problem='vectors'):
        self.writer= SummaryWriter(logdir)
        self.loss=Loss
        self.problem=problem
        self.metric=metric
        self.dict={}
    
    
    #---------------------------------------------- Vectors & matrices
    #------------------------------------------------------------------
    @eqx.filter_jit
    def loss_fn_class(self, params, statics, x, y):
        model = eqx.combine(params, statics)
        pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
        return -jnp.mean(y * pred_y)

    @eqx.filter_jit
    def loss_fn_mse(self, params, statics, x, y):
        model = eqx.combine(params, statics)
        return jnp.mean((y - jax.vmap(model)(x))**2)
    
    @eqx.filter_jit
    def accuracy_vectors(self,params, statics, x, y):
        model = eqx.combine(params, statics)
        pred = jnp.argmax( jax.nn.softmax(jax.vmap(model)(x)), axis=1) 
        y = jnp.argmax( y, axis=1) 
        return jnp.mean(pred == y)
    
    @eqx.filter_jit
    def mse_vectors(self,params, statics,  x, y):
        model = eqx.combine(params, statics)
        return jnp.mean( (y-jax.vmap(model)(x))**2 )
    
    
    #------------------------------------------------------------ Graphs 
    #-------------------------------------------------------------------
    @eqx.filter_jit
    def loss_fn_class_graph(self, params, statics, x, y, adj=None):
        model = eqx.combine(params, statics)
        logits = model(x, adj)
        # print(logits.shape, y.shape)
        # print("the logits shape", logits.shape)
        return -jnp.mean(y * jax.nn.log_softmax(logits))

    @eqx.filter_jit
    def loss_fn_mse_graph(self, params, statics, x, y, adj=None):
        model = eqx.combine(params, statics)
        return jnp.mean((y - jax.vmap(model(x, adj)))**2)
    
    @eqx.filter_jit
    def accuracy_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return model(x, adj)
    @eqx.filter_jit
    def accuracy_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return jnp.mean(jnp.argmax(model(x, adj), axis=1) == jnp.argmax(y, axis=1))
    
    @eqx.filter_jit
    def mse_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return jnp.mean((jax.vmap(model)(x, adj) - y)**2 )
    

    # ------------------------------------------------------------ Graphs
    # -------------------------------------------------------------------

    @eqx.filter_jit
    def get_pred(self, params, statics, x):
        model = eqx.combine(params, statics)
        return jax.vmap(model)(x)

    @eqx.filter_jit
    def get_pred_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return jax.vmap(model)(x, adj)


    # -------------------------------------------------------------------
    def return_loss_grad(self, params, batch, static):
            if self.problem=='vectors':
                (x, y) = batch
                if self.loss == 'class':
                    grads =jax.grad(self.loss_fn_class)(params, static, x, y)
                    loss = self.loss_fn_class(params, static, x, y)
                elif self.loss=='mse':
                    grads= jax.grad(self.loss_fn_mse)(params, static, x, y)
                    loss  =self.loss_fn_mse(params, static, x, y)
            elif self.problem== 'graphs':
                # print("I came here and wemnt to calculate the loss")
                (x, y, adj) = batch
                if self.loss == 'class':
                    grads  =jax.grad(self.loss_fn_class_graph)(params, static, x, y, adj=adj)
                    loss = self.loss_fn_class_graph(params, static, x, y, adj=adj)
                elif self.loss=='mse':
                    grads  =jax.grad(self.loss_fn_mse_graph)(params, static, x, y, adj=adj)
                    loss =  self.loss_fn_mse_graph(params, static, x, y, adj=adj)
            return loss, grads
    

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    def train__EUC__(self, trainloader, valloader, params, static, optim, n_iter=1000, save_iter=10):
        trainiter = iter(trainloader)
        batch = next(trainiter)
        x, y = batch
        x = x.numpy().astype(np_.float32)
        y = y.numpy().astype(np_.float32)
        batch = (x, y)
        opt_state = optim.init_state(params, batch=batch, static=static)
        from tqdm import tqdm 
        pbar = tqdm(range(n_iter))
        train_L=[]
        grads=[]
        for step in pbar:
            try:
                batch = next(trainiter)
            except StopIteration:
                trainiter = iter(trainloader)
                batch = next(trainiter)
            (x, y) = batch
            x = x.numpy().astype(np_.float32)
            y = y.numpy().astype(np_.float32)
            batch = (x, y)
            params, opt_state  =  optim.update(
                                  params=params, state=opt_state,  batch=batch, static=static)
            l, _, _, g = self.evaluate__(step, batch, params, static)
            train_L.append(l.item())
            grads.append(g.item())
            pbar.set_postfix({"Loss:": sum(train_L)/len(train_L)})
            self.writer.add_scalar('Loss/train', l.item(), step)
            self.writer.add_scalar('gradient/train', g.item(), step)
            if step % save_iter == 0:
                score = []
                L = []
                # --------------------------------------------------------
                for batch in valloader:
                    (x,y) = batch
                    x = x.numpy().astype(np_.float32)
                    y = y.numpy().astype(np_.float32)
                    loss, score__, _, _ = self.evaluate__(0, (x, y), params, static)
                    score.append(score__.item())
                    L.append(loss.item())
                self.dict[str(step)] = (
                    score,  L, train_L, grads, copy.deepcopy(params), copy.deepcopy(opt_state))


                self.writer.add_scalar('Loss/validation', sum(L)/len(L), step)
                self.writer.add_scalar('score/validation', sum(score)/len(score), step)
                self.dict['params']=params
                self.dict['static']=static

        self.writer.flush()
        with open('ckpt.pkl', 'wb') as f:
            pickle.dump(self.dict, f)
        return params, static, optim
        
    
    def evaluate__(self, epoch, batch, params, static):
        
        # --- Get Loss
        if self.problem=='vectors':
            (x, y) = batch
            if self.loss == 'class':
                loss = self.loss_fn_class(params, static, x, y)
            elif self.loss=='mse':
                loss, grads = self.return_loss_grad(params, (x,y), static)
                grads = jax.tree_util.tree_leaves(grads)
                grads = jnp.mean(jnp.asarray([jnp.linalg.norm(g) for g in grads]))
            
        elif self.problem== 'graphs':
            (x, y, adj) = batch
            if self.loss == 'class':
                loss  =self.loss_fn_class_graph(params, static, x, y, adj=adj)
            elif self.loss=='mse':
                loss  =self.loss_fn_mse_graph(params, static, x, y, adj=adj)

        # --- Get score
        if self.problem == 'vectors':
            (x, y) = batch
            if self.loss == 'class':
                score =self.accuracy_vectors(params, static, x, y)
            elif self.loss=='mse':
                score =self.mse_vectors(params, static, x, y)
        elif self.problem== 'graphs':
            (x, y, adj) = batch
            if self.loss == 'class':
                score = self.accuracy_graphs(params, static, x, y, adj=adj)
            elif self.loss=='mse':
                score =self.accuracy_graphs(params, static, x, y, adj=adj)
                
        # --- Get prediction
        if self.problem=='vectors':
            (x, _) = batch
            pred= self.get_pred(params, static, x)
        if self.problem == 'graphs':
            (x, _) = batch
            pred = self.get_pred(params, static, x)
                                
        
        return loss, score, pred, grads





    def writer(self, dict, epoch, string_scalers= ['train'], metric_scaler=['training_loss', 'validation_loss', 'loss', 'acc']):
        for (string, metric) in zip(string_scalers, metric_scaler):
            self.writer.add_scalar(str(string), dict[metric], epoch)
        pickle.dump( dict['params'], open("best_ckpt.pkl"), "wb")







    def train__NONEUC__(self, trainloader, params, static, optim, n_iter=1000):

        # print(x)
        # print(y)
        # x = x.numpy().astype(np_.float32)
        # adj = adj.numpy().astype(np_.float32)
        # y = jax.nn.one_hot(jnp.array(y.astype(np_.int32)), 7)
        (x, y, adj) = trainloader
        opt_state = optim.init_state(params, batch=(x, y, adj), static=static)
        loss__ = []
        score__ = []
        from tqdm import tqdm
        pbar = tqdm(range(n_iter))
        for step in pbar:
            (x, y, adj) = trainloader
            params, opt_state = optim. update(
                params=params, state=opt_state,  batch=(x, y, adj), static=static)
            # trainer.train__(step, batch, params, static, optim, opt_state)
            # params , static = eqx.partition(model, eqx.is_array)
            l, s = self.evaluate__(step, (x, y, adj), params, static)
            loss__.append(l)
            score__.append(s)
            pbar.set_postfix({"Loss:": sum(loss__)/len(loss__),
                             "accuracy": sum(score__)/len(score__)})


