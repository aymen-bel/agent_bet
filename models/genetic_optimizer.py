# genetic_optimizer.py
import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
import os

class GeneticWeightOptimizer:
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.15, 
                 crossover_rate: float = 0.85, elite_size: int = 10,
                 tournament_size: int = 5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.generation = 0
        
        # تعريف فضاء البحث للأوزان
        self.weight_bounds = {
            'points_per_match': (0.05, 0.25),
            'win_rate': (0.05, 0.20),
            'goal_difference': (0.03, 0.15),
            'goals_per_match': (0.03, 0.12),
            'conceded_per_match': (0.03, 0.12),
            'shot_efficiency': (0.02, 0.10),
            'conversion_rate': (0.02, 0.10),
            'attacking_pressure': (0.01, 0.08),
            'attacking_consistency': (0.01, 0.08),
            'big_chances': (0.01, 0.06),
            'defensive_efficiency': (0.04, 0.18),
            'clean_sheet_rate': (0.02, 0.10),
            'defensive_consistency': (0.01, 0.08),
            'pressure_resistance': (0.01, 0.06),
            'current_form': (0.03, 0.12),
            'form_momentum': (0.01, 0.08),
            'consistency_score': (0.02, 0.10),
            'performance_trend': (0.01, 0.06),
            'home_advantage': (0.01, 0.08),
            'away_resilience': (0.01, 0.08),
            'motivation_factor': (0.01, 0.06),
            'external_factor': (0.005, 0.04),
            'opponent_strength': (0.01, 0.08),
            'match_importance': (0.005, 0.04),
            'pressure_handling': (0.005, 0.03)
        }
        
        self.weight_keys = list(self.weight_bounds.keys())
        
        # إعداد التسجيل
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # إنشاء مجلد الإخراج
        self.output_dir = "output/training"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def initialize_population(self) -> List[Dict[str, float]]:
        """تهيئة المجتمع الأولي مع تنوع كبير"""
        population = []
        
        for i in range(self.population_size):
            individual = {}
            
            # إنشاء فرد مع تنوع في استراتيجية الأوزان
            strategy = random.choice(['balanced', 'attack_focused', 'defense_focused', 'form_focused'])
            
            for key, (min_val, max_val) in self.weight_bounds.items():
                if strategy == 'attack_focused' and 'attack' in key:
                    individual[key] = random.uniform(min_val * 1.2, max_val)
                elif strategy == 'defense_focused' and 'defens' in key:
                    individual[key] = random.uniform(min_val * 1.2, max_val)
                elif strategy == 'form_focused' and 'form' in key:
                    individual[key] = random.uniform(min_val * 1.2, max_val)
                else:
                    individual[key] = random.uniform(min_val, max_val)
            
            # تطبيع الأوزان
            population.append(self._normalize_weights(individual))
        
        self.population = population
        self.logger.info(f"🎯 تم تهيئة مجتمع من {len(population)} فرد")
        return population
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """تطبيع الأوزان بحيث يكون مجموعها 1"""
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0/len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}
    
    def tournament_selection(self, fitness_scores: List[float]) -> List[Dict[str, float]]:
        """اختيار الآباء باستخدام طريقة البطولة"""
        parents = []
        
        for _ in range(self.population_size - self.elite_size):
            # اختيار عشوائي للمتنافسين
            competitors = random.sample(list(zip(self.population, fitness_scores)), 
                                      self.tournament_size)
            # اختيار الأفضل
            best_individual = max(competitors, key=lambda x: x[1])[0]
            parents.append(best_individual)
        
        return parents
    
    def blended_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """تكاثر مختلط (Blended Crossover)"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = {}
        child2 = {}
        alpha = random.uniform(0.3, 0.7)  # معلمة الاختلاط
        
        for key in self.weight_keys:
            p1_val = parent1[key]
            p2_val = parent2[key]
            
            # تكاثر مختلط
            child1[key] = alpha * p1_val + (1 - alpha) * p2_val
            child2[key] = (1 - alpha) * p1_val + alpha * p2_val
        
        return self._normalize_weights(child1), self._normalize_weights(child2)
    
    def adaptive_mutation(self, individual: Dict[str, float], generation: int, max_generations: int) -> Dict[str, float]:
        """طفرة تكيفية تقل مع تقدم الأجيال"""
        mutated = individual.copy()
        current_mutation_rate = self.mutation_rate * (1 - generation / max_generations)
        
        for key in self.weight_keys:
            if random.random() < current_mutation_rate:
                # طفرة غاوسية
                min_val, max_val = self.weight_bounds[key]
                current_val = mutated[key]
                
                # انحراف معياري يتناقص مع الأجيال
                std_dev = (max_val - min_val) * 0.1 * (1 - generation / max_generations)
                mutation = np.random.normal(0, std_dev)
                
                new_val = current_val + mutation
                new_val = max(min_val, min(max_val, new_val))
                mutated[key] = new_val
        
        return self._normalize_weights(mutated)
    
    def evaluate_individual(self, individual: Dict, training_data: List[Dict], 
                          fitness_function: Callable) -> float:
        """تقييم لياقة فرد واحد"""
        try:
            return fitness_function(individual, training_data)
        except Exception as e:
            self.logger.warning(f"⚠️  خطأ في تقييم الفرد: {e}")
            return 0.0
    
    def evaluate_population_parallel(self, training_data: List[Dict], 
                                   fitness_function: Callable) -> List[float]:
        """تقييم لياقة المجتمع بالتوازي"""
        fitness_scores = [0.0] * len(self.population)
        
        # استخدام multiprocessing للتقييم المتوازي
        with ProcessPoolExecutor(max_workers=min(8, len(self.population))) as executor:
            future_to_index = {
                executor.submit(self.evaluate_individual, ind, training_data, fitness_function): i
                for i, ind in enumerate(self.population)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    fitness_scores[index] = future.result()
                except Exception as e:
                    self.logger.error(f"❌ فشل تقييم الفرد {index}: {e}")
                    fitness_scores[index] = 0.0
        
        return fitness_scores
    
    def create_diverse_generation(self, parents: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """إنشاء جيل جديد مع الحفاظ على التنوع"""
        new_population = []
        
        # الحفاظ على النخبة
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # التكاثر لإكمال المجتمع
        while len(new_population) < self.population_size:
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.blended_crossover(parent1, parent2)
                
                # طفرة تكيفية
                child1 = self.adaptive_mutation(child1, self.generation, 100)
                child2 = self.adaptive_mutation(child2, self.generation, 100)
                
                new_population.extend([child1, child2])
            else:
                # احتياطي: أفراد عشوائيون
                new_population.append(self._create_random_individual())
        
        return new_population[:self.population_size]
    
    def _create_random_individual(self) -> Dict[str, float]:
        """إنشاء فرد عشوائي جديد"""
        individual = {}
        for key, (min_val, max_val) in self.weight_bounds.items():
            individual[key] = random.uniform(min_val, max_val)
        return self._normalize_weights(individual)
    
    def evolve(self, training_data: List[Dict], fitness_function: Callable, 
               generations: int = 100, early_stopping: int = 25,
               callback: Optional[Callable] = None) -> Dict[str, float]:
        """تطور المجتمع عبر الأجيال"""
        # تهيئة المجتمع
        if not self.population:
            self.initialize_population()
        
        best_fitness_history = []
        no_improvement_count = 0
        
        for generation in range(generations):
            self.generation = generation
            
            # تقييم اللياقة
            fitness_scores = self.evaluate_population_parallel(training_data, fitness_function)
            
            # تحديث أفضل فرد
            current_best_fitness = max(fitness_scores)
            current_best_index = np.argmax(fitness_scores)
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = self.population[current_best_index].copy()
                no_improvement_count = 0
                
                self.logger.info(f"🚀 الجيل {generation}: أفضل لياقة = {current_best_fitness:.4f}")
                
                # حفظ أفضل فرد
                self._save_best_individual(generation)
            else:
                no_improvement_count += 1
            
            best_fitness_history.append(current_best_fitness)
            self.fitness_history.append(np.mean(fitness_scores))
            
            # استدعاء callback إذا كان موجوداً
            if callback:
                population_stats = {
                    'population_size': self.population_size,
                    'mutation_rate': self.mutation_rate,
                    'crossover_rate': self.crossover_rate,
                    'elite_size': self.elite_size
                }
                callback(generation, current_best_fitness, np.mean(fitness_scores), 
                        population_stats, self.best_individual)
            
            # تسجيل التقدم
            if generation % 10 == 0:
                avg_fitness = np.mean(fitness_scores)
                fitness_std = np.std(fitness_scores)
                self.logger.info(
                    f"📊 الجيل {generation}: "
                    f"متوسط لياقة = {avg_fitness:.4f} (±{fitness_std:.4f}), "
                    f"أفضل = {current_best_fitness:.4f}"
                )
            
            # التحقق من التوقف المبكر
            if no_improvement_count >= early_stopping and generation >= 50:
                self.logger.info(f"🛑 توقف مبكر عند الجيل {generation}")
                break
            
            # اختيار الآباء
            parents = self.tournament_selection(fitness_scores)
            
            # إنشاء جيل جديد
            self.population = self.create_diverse_generation(parents, fitness_scores)
        
        self.logger.info(f"✅ اكتمل التطور بعد {self.generation} جيل! أفضل لياقة: {self.best_fitness:.4f}")
        
        # حفظ تاريخ التطور النهائي
        self.save_evolution_history()
        
        return self.best_individual
    
    def _save_best_individual(self, generation: int):
        """حفظ أفضل فرد في الملف"""
        best_data = {
            'generation': generation,
            'fitness': self.best_fitness,
            'weights': self.best_individual,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.output_dir}/best_genetic_individual_gen_{generation}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(best_data, f, ensure_ascii=False, indent=2)
    
    def save_evolution_history(self):
        """حفظ تاريخ التطور الكامل"""
        history = {
            'fitness_history': self.fitness_history,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'total_generations': self.generation,
            'parameters': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size
            },
            'weight_bounds': self.weight_bounds,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.output_dir}/genetic_evolution_history.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 تم حفظ تاريخ التطور في: {filename}")
    
    def get_best_weights(self) -> Dict[str, float]:
        """الحصول على أفضل الأوزان"""
        if self.best_individual is not None:
            return self.best_individual
        else:
            self.logger.warning("⚠️  لا يوجد أفضل فرد، إرجاع أوزان متوازنة")
            return self.get_balanced_weights()
    
    def get_balanced_weights(self) -> Dict[str, float]:
        """الحصول على أوزان متوازنة (بديل)"""
        balanced = {}
        for key, (min_val, max_val) in self.weight_bounds.items():
            balanced[key] = (min_val + max_val) / 2
        return self._normalize_weights(balanced)
    
    def analyze_weight_evolution(self) -> Dict[str, List[float]]:
        """تحليل تطور الأوزان عبر الأجيال"""
        if not self.best_individual:
            return {}
        
        analysis = {}
        for key in self.weight_keys:
            analysis[key] = [self.best_individual[key]]
        
        return analysis