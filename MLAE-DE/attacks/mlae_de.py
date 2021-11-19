import numpy as np
import logging
import torch
import gc
from multiprocessing import Pool
from ml_liw_model.train  import criterion
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
class MLDE(object):
    def __init__(self, model):
        self.model = model

    def generate_np(self, x_list, **kwargs):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logging.info('prepare attack')
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        eps = kwargs['eps']
        pop_size = kwargs['pop_size']
        generation = kwargs['generation']
        batch_size = kwargs['batch_size']
        x_adv = []
        success = 0
        nchannels,img_rows, img_cols,  = x_list.shape[1:4]
        count = 0
        for i in range(len(x_list)):
                target_label = np.argwhere(y_target[i] > 0)
                r, count_tem, _2 = DE(pop_size, generation, img_rows * img_cols * nchannels, self.model, x_list[i],
                                   target_label, eps, batch_size, gradient=None)
                x_adv_tem = np.clip(x_list[i] + np.reshape(r, x_list.shape[1:]) * eps, 0, 1)
                count += count_tem
                with torch.no_grad():
                    if torch.cuda.is_available():
                        adv_pred = self.model(
                            torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32).cuda()).cpu()
                    else:
                        adv_pred = self.model(torch.tensor(np.expand_dims(x_adv_tem, axis=0), dtype=torch.float32))
                adv_pred = np.asarray(adv_pred)
                pred = adv_pred.copy()
                pred[pred >= (0.5 + 0)] = 1
                pred[pred < (0.5 + 0)] = -1
                adv_pred_match_target = np.all((pred == y_target[i]), axis=1)
                if adv_pred_match_target:
                    success = success + 1
                x_adv.append(x_adv_tem)
                logging.info('Successfully generated adversarial examples on '+str(success)+' of '+str(batch_size)+' instances')

        return x_adv , count

class Problem:
    def __init__(self, model, image, target_label, eps, batch_size):
        self.model = model
        self.image = image
        self.target_label = target_label
        self.eps = eps
        self.batch_size = batch_size


    def evaluate(self, x):
        with torch.no_grad():
            if torch.cuda.is_available():
                predict = self.model(torch.tensor(np.clip(np.tile(self.image, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.,
                                                          1.), dtype=torch.float32).cuda()).cpu()
            else:
                predict = self.model(torch.tensor(np.clip(np.tile(self.image, (len(x), 1, 1, 1))
                                                          + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.,
                                                          1.), dtype=torch.float32))
        p = np.copy(predict)
        q = np.zeros(p.shape)+0.5
        fit = p-q
        fit[:,self.target_label]=-fit[:,self.target_label]
        fit[np.where(fit<0)]=0
        fitness = np.sum(fit,axis=1)
        fitness = fitness[:, np.newaxis]
        return fitness, fit

def mating(pop,F):
    p2 = np.copy(pop)
    np.random.shuffle(p2)
    p3 = np.copy(p2)
    np.random.shuffle(p3)
    mutation = pop + F * (p2 - p3)
    return mutation

def select(pop,fitness,fit,off,off_fitness,off_fit):
   new_pop = pop.copy()
   new_fitness = fitness.copy()
   new_fit = fit.copy()
   i=np.argwhere(fitness>off_fitness)
   new_pop[i] = off[i].copy()
   new_fitness[i] = off_fitness[i].copy()
   new_fit[i] = off_fit[i].copy()
   return new_pop ,new_fitness ,new_fit

def complement (fit,pop, fitness,problem) :
    popnew = pop.copy()
    sort = np.argsort(fitness.reshape(-1))
    for q in range (len(pop)):
        i = sort[q]
        fit_item = fit.copy()
        c = np.argwhere(fit[i] == 0)
        fit_item[:, c] = 0
        fitness_tem = np.sum(fit_item, axis=1)
        j = np.argmin(fitness_tem)
        popnew[i] = pop[i] + pop[j]*0.5
    off_fitness_new, off_fit_new = problem.evaluate(popnew)
    pop1, fitness1, fit1 = select(pop, fitness, fit, popnew, off_fitness_new, off_fit_new)
    return pop1,fitness1, fit1


def DE(pop_size, generation, length, model, image, target_label, eps, batch_size, gradient):
    generation_save = np.zeros((10000,))
    problem = Problem(model, image, target_label, eps, batch_size)
    pop = np.random.uniform(-1, 1, size=(pop_size, length))
    if (not (gradient is None)):
        pop[0] = np.reshape(np.sign(gradient), (length))
    max_eval = pop_size * generation
    eval_count = 0
    fitness, fit = problem.evaluate(pop)
    eval_count += pop_size
    count = 0
    fitmin = np.min(fitness)
    generation_save[count] = fitmin
    F = 0.5
    if (len(np.where(fitness == 0)[0]) == 0):
        while (eval_count < max_eval):
            count += 1
            off = mating(pop,F)
            off_fitness , off_fit = problem.evaluate(off)
            eval_count += pop_size
            pop ,fitness ,fit = select (pop,fitness,fit,off,off_fitness,off_fit)
            pop, fitness, fit = complement(fit, pop, fitness, problem)
            fitmin = np.min(fitness)
            generation_save[count] = fitmin
            if (len(np.where(fitness == 0)[0]) != 0):
                break
    if (len(np.where(fitness == 0)[0]) != 0):
        return pop[np.where(fitness == 0)[0][0]], eval_count, generation_save[:count + 1]
    else:
        return pop[0], eval_count/100, generation_save[:count + 1]
