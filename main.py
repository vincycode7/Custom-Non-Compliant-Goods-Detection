from torch.nn import functional as F
from GA_optim import *
from model import *

if __name__ == "__main__":

    #Initialize the GA Optimizer with the model class
    # fitness =     def my_loss(self, y_hat, y):
    #     return F.cross_entropy(y_hat, y)
    fitness_func = lambda y_hat_,y: F.cross_entropy(y_hat, y)
    model_instance = Custom_Model
    model_name = 'twoclasses_model1_ga.pt'
    gen_algo = GA_optim(    
                            model_object=model_instance, 
                            Npop=6, 
                            num_generations=2,
                            epoch_per_model=1,
                            check_val_every_n_epoch = 1,
                            turn_off_backpropagation=False, 
                            num_parents_mating=3,
                            parent_selection = True,
                            crossover=True,
                            crossover_probability=0.5,
                            n_multi_crossover=6,
                            keep_parents=True, 
                            mutation=True,
                            mutation_probability=0.05,
                            mutation_num_chromes=3,
                            mutation_num_genes=20,
                            save_every_n_gen=1,
                            model_name=model_name,
                            to_gpu=0
                        )

    # gen_algo.from_state(path_to_state_dict=model_name)
    try:
        print(f'{model_name}')
        gen_algo.from_state(path_to_state_dict=model_name)
        print(f'loading Succesful')
    except Exception as e:
        print(f"Warning!!! Error Occured!!! while loading Model. Error is --> {e}")
        
    gen_algo.train(generation=10)
    gen_algo.save_state(name_of_state=model_name)