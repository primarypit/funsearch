from __future__ import annotations
import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping
from queue import PriorityQueue
from absl import logging
import numpy as np
import scipy
from tqdm import tqdm
from implementation import code_manipulation
from implementation import config as config_lib
from nltk.metrics import edit_distance
import json

ScoresPerTest = Mapping[Any, float]

def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """Reduces per-test scores into a single score.
    """
    # TODO RZ: change the code to average the score of each test.
    # return scores_per_test[list(scores_per_test.keys())[-1]]
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)

@dataclasses.dataclass(frozen=True)
class Prompt_NS:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated implementation
         into the same island.
    """
    code: str
    version_generated: int


class Program_NS():
    def __init__(self, score: float, routes: list, implementation: code_manipulation.Function) -> None:
        self.score = score
        self.routes = routes
        self.implementation = implementation
    
    def get_rotues(self) -> list:
        return self.routes
    def get_score(self) -> float:
        return self.score
    def get_imp(self) -> code_manipulation.Function:
        return self.implementation

class ProgramsDatabase_NS():
    def __init__(
            self,
            config: config_lib.ProgramsDatabase_NS_Config,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        self.k = self._config.k
        self.threshold = self._config.threshold
        self.volume = self._config.volume
        self._functions_per_prompt = self._config.functions_per_prompt
        self.pop: list[Program_NS] = []
        self.bestscore = 0
        self.bestprogram: Program_NS = None
        self.reset_num = 0
        self.register_num = 0
        self.save_period = 10
        self.score_threshold = 0.9
        self.gamma = 1.5
    
    def calc_sim(self, routes1, routes2):
        # 0 <= sum / l < 1, smaller value -> more similar
        l = len(routes1)
        sum = 0
        for i in range(l):
            r1, r2 = routes1[i], routes2[i]
            sum += edit_distance(r1, r2) / len(r1)
        return sum / l

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
            routes: list,
            **kwargs
    ) -> None:
        
        check_flag = False
        score = _reduce_score(scores_per_test)
        cur_program = program
        if score > self.bestscore:
            logging.info('Best score increased to %s', score)
        
        # 0 < priority <= 1
        if self.pop == []:
            Cur_P = Program_NS(score, routes, cur_program)
            self.bestscore = score
            self.bestprogram = Cur_P
            self.pop.append(Cur_P)
            self.register_num += 1
            check_flag = True
        elif score > self.bestscore: # new best program
            Cur_P = Program_NS(score, routes, cur_program)
            self.bestscore = score
            self.bestprogram = Cur_P
            self.pop.append(Cur_P)
            self.register_num += 1
            check_flag = True
        else:
            sims = [] # all sim belong to (0, 1)
            
            for P in self.pop:
                target_routes = P.get_rotues()
                sims.append(self.calc_sim(routes, target_routes))
            novelty_v = sum(sorted(sims)[:self.k]) / self.k
            if novelty_v  > self.threshold and score > self.score_threshold:
                logging.info("Current novelty %s, Current score %s, Accept.", novelty_v, score)
                Cur_P = Program_NS(score, routes, cur_program)
                self.pop.append(Cur_P)
                self.register_num += 1
                check_flag = True
            else:
                logging.info("Current novelty %s, Current score %s, Reject", novelty_v, score)
        
        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            cur_program.score = score
            cur_program.global_sample_nums = global_sample_nums
            cur_program.sample_time = sample_time
            cur_program.evaluate_time = evaluate_time
            profiler.register_function_NS(cur_program, check_flag)

        if len(self.pop) > self.volume:
            logging.info("Reset...")
            self.pop_pop()

        if self.register_num % self.save_period == 0:
            self.threshold *= self.gamma
            if self.threshold > 1:
                self.threshold = 1 / self.threshold
            self.save_programs_after_reset()
            logging.info("New threshold %s", self.threshold)

    def pop_pop(self):

        rous = []

        tmp_best_P = None
        best_id = None
        tmp_best_score = 0
        for i in range(len(self.pop)):
            cur_score = self.pop[i].get_score()
            if cur_score > tmp_best_score:
                tmp_best_score = cur_score
                best_id = i
        
        tmp_best_P = self.pop.pop(best_id)

        pbar = tqdm(list(range(len(self.pop))))
        for i in pbar:
            pbar.set_description("Processing pop " + i)
            routes = self.pop[i].get_rotues()
            sims = []
            for j in range(len(self.pop)):
                if j != i:
                    target_routes = self.pop[j].get_rotues()
                    sims.append(self.calc_sim(routes, target_routes))
            rous.append(sum(sorted(sims)[:self.k]) / self.k)
        
        keep_ids = np.argsort(-rous)[:self.volume - 1]

        old_pop = self.pop
        self.pop = []
        self.pop.append(tmp_best_P)

        for id in keep_ids:
            self.pop.append(old_pop[id])

        del old_pop
    
    def get_prompt(self) -> Prompt_NS:
        
        functions_per_prompt = min(len(self.pop), self._functions_per_prompt)

        idx = np.random.choice(len(self.pop), size=functions_per_prompt)

        implementations = []
        scores = []
        for id in idx:
            implementations.append(self.pop[id].get_imp())
            scores.append(self.pop[id].get_score())

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1

        return Prompt_NS(self._generate_prompt(sorted_implementations), version_generated)

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """Creates a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)  # We will mutate these.

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        # Create the header of the function to be generated by the LLM.
        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        # Replace functions in the template with the list constructed here.
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)
    
    def save_programs_after_reset(self):
        curprograms = [str(P.get_imp()) for P in self.pop]
        with open("programs_round_{}.json".format(self.register_num), "w") as file:
            json.dump(curprograms, file)

    def save_allprograms(self):
        allprograms = [str(P.get_imp()) for P in self.pop]
        with open("allprograms.json", "w") as file:
            json.dump(allprograms, file)
        with open("bestprograms.json", "w") as file:
            json.dump(str(self.bestprogram.get_imp()))