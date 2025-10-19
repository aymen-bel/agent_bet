# smart_optimizer.py
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Callable
import random
from datetime import datetime
import logging
import os

class SmartWeightOptimizer:
    def __init__(self, state_dim: int = 20, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.weights = self.initialize_weights()
        self.weights_history = []
        self.performance_history = []
        self.best_weights = self.weights.copy()
        self.best_reward = -np.inf
        
        # عوامل التعلم
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.995
        
        # ذاكرة التعلم
        self.memory = []
        self.memory_size = 1000
        
        # إعداد التسجيل
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # إنشاء مجلد الإخراج
        self.output_dir = "output/reinforcement_learning"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/episodes", exist_ok=True)
    
    def initialize_weights(self) -> Dict[str, float]:
        """تهيئة الأوزان بشكل عشوائي مع قيود واقعية"""
        return {
            # أوزان الأداء الأساسي
            'points_per_match': np.random.uniform(0.1, 0.3),
            'win_rate': np.random.uniform(0.1, 0.25),
            'goal_difference': np.random.uniform(0.05, 0.15),
            
            # أوزان الهجوم
            'shot_efficiency': np.random.uniform(0.05, 0.1),
            'conversion_rate': np.random.uniform(0.05, 0.1),
            'attacking_pressure': np.random.uniform(0.03, 0.08),
            
            # أوزان الدفاع
            'defensive_efficiency': np.random.uniform(0.05, 0.15),
            'clean_sheet_rate': np.random.uniform(0.03, 0.08),
            'goals_conceded_per_match': np.random.uniform(0.05, 0.12),
            
            # أوزان الاتساق والشكل
            'consistency_score': np.random.uniform(0.03, 0.08),
            'current_form': np.random.uniform(0.05, 0.1),
            'form_momentum': np.random.uniform(0.02, 0.06),
            
            # أوزان العوامل الخارجية
            'motivation_factor': np.random.uniform(0.02, 0.06),
            'external_factor': np.random.uniform(0.01, 0.04),
            'home_advantage': np.random.uniform(0.03, 0.08),
            
            # أوزان إضافية
            'opponent_strength': np.random.uniform(0.02, 0.06),
            'performance_trend': np.random.uniform(0.01, 0.04),
            'pressure_handling': np.random.uniform(0.01, 0.03),
            'attacking_consistency': np.random.uniform(0.02, 0.05),
            'defensive_consistency': np.random.uniform(0.02, 0.05)
        }
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """تطبيع الأوزان بحيث يكون مجموعها 1"""
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}
    
    def get_state_features(self, team_metrics: Dict, opponent_metrics: Dict, context: Dict) -> np.ndarray:
        """استخراج سمات الدولة للتعلم المعزز"""
        features = []
        
        # سمات الأداء النسبي
        features.append(team_metrics.get('points_per_match', 0))
        features.append(opponent_metrics.get('points_per_match', 0))
        features.append(team_metrics.get('win_rate', 0) - opponent_metrics.get('win_rate', 0))
        
        # سمات الهجوم والدفاع
        features.append(team_metrics.get('goals_per_match', 0))
        features.append(opponent_metrics.get('goals_conceded_per_match', 0))
        features.append(team_metrics.get('defensive_efficiency', 0))
        features.append(opponent_metrics.get('shot_efficiency', 0))
        
        # سمات الشكل والاتساق
        features.append(team_metrics.get('current_form', 0))
        features.append(opponent_metrics.get('current_form', 0))
        features.append(team_metrics.get('consistency_score', 0))
        
        # سمات العوامل الخارجية
        features.append(team_metrics.get('motivation_factor', 1.0))
        features.append(opponent_metrics.get('motivation_factor', 1.0))
        features.append(context.get('home_advantage', 1.0))
        features.append(context.get('importance_multiplier', 1.0))
        
        # سمات إضافية
        features.append(team_metrics.get('performance_trend', 0))
        features.append(team_metrics.get('attacking_consistency', 0))
        features.append(team_metrics.get('defensive_consistency', 0))
        
        # ملء السمات المتبقية بقيم افتراضية
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features[:self.state_dim])
    
    def select_action(self, state: np.ndarray) -> Dict[str, float]:
        """اختيار action (تعديل الأوزان) باستخدام استراتيجية ε-greedy"""
        if random.random() < self.exploration_rate:
            # استكشاف: تعديل عشوائي للأوزان
            return self.explore_weights()
        else:
            # استغلال: تعديل الأوزان بناءً على التعلم السابق
            return self.exploit_weights(state)
    
    def explore_weights(self) -> Dict[str, float]:
        """استكشاف: تعديل عشوائي للأوزان"""
        new_weights = self.weights.copy()
        
        # تعديل عشوائي لبعض الأوزان
        weights_to_modify = random.sample(list(new_weights.keys()), 
                                        k=random.randint(3, 8))
        
        for weight_key in weights_to_modify:
            modification = np.random.normal(0, 0.1)  # تعديل صغير
            new_weights[weight_key] = max(0.01, new_weights[weight_key] + modification)
        
        return self.normalize_weights(new_weights)
    
    def exploit_weights(self, state: np.ndarray) -> Dict[str, float]:
        """استغلال: تعديل الأوزان بناءً على التعلم"""
        new_weights = self.weights.copy()
        
        # تحليل الدولة لتحديد الأوزان التي تحتاج تعديل
        state_analysis = self.analyze_state(state)
        
        for weight_key, adjustment in state_analysis.items():
            new_weights[weight_key] = max(0.01, new_weights[weight_key] + adjustment)
        
        return self.normalize_weights(new_weights)
    
    def analyze_state(self, state: np.ndarray) -> Dict[str, float]:
        """تحليل الدولة لتحديد تعديلات الأوزان المناسبة"""
        adjustments = {}
        
        # تحليل بسيط بناءً على قيم الدولة
        if state[0] > 0.6:  # نقاط عالية
            adjustments['points_per_match'] = 0.05
        if state[2] > 0.2:  # فرق فوز كبير
            adjustments['win_rate'] = 0.03
        if state[4] < 1.0:  # دفاع ضعيف للخصم
            adjustments['shot_efficiency'] = 0.02
        if state[10] > 1.1:  # تحفيز عالي
            adjustments['motivation_factor'] = 0.01
        
        return adjustments
    
    def calculate_reward(self, predictions: List[Dict], actual_results: Dict) -> float:
        """حساب المكافأة بناءً على دقة التنبؤات"""
        if not predictions or not actual_results:
            return 0.0
        
        reward = 0.0
        total_weight = 0
        
        # مكافأة دقة توقع النتيجة
        result_accuracy = self.calculate_result_accuracy(predictions, actual_results)
        reward += result_accuracy * 0.4
        total_weight += 0.4
        
        # مكافأة دقة توقع الأهداف
        score_accuracy = self.calculate_score_accuracy(predictions, actual_results)
        reward += score_accuracy * 0.3
        total_weight += 0.3
        
        # مكافأة دقة توقع الفارق
        diff_accuracy = self.calculate_difference_accuracy(predictions, actual_results)
        reward += diff_accuracy * 0.2
        total_weight += 0.2
        
        # مكافأة الثقة (عقاب للثقة العالية في التنبؤات الخاطئة)
        confidence_penalty = self.calculate_confidence_penalty(predictions, actual_results)
        reward -= confidence_penalty * 0.1
        total_weight += 0.1
        
        if total_weight > 0:
            reward /= total_weight
        
        return max(0.0, reward)
    
    def calculate_result_accuracy(self, predictions: List[Dict], actual: Dict) -> float:
        """حساب دقة توقع النتيجة (فوز/تعادل/خسارة)"""
        if not predictions:
            return 0.0
        
        best_pred = predictions[0]
        pred_home = best_pred.get('home_goals', 0)
        pred_away = best_pred.get('away_goals', 0)
        actual_home = actual.get('home_goals', 0)
        actual_away = actual.get('away_goals', 0)
        
        # تحديد النتيجة المتوقعة والفعلية
        pred_result = 'H' if pred_home > pred_away else 'A' if pred_away > pred_home else 'D'
        actual_result = 'H' if actual_home > actual_away else 'A' if actual_away > actual_home else 'D'
        
        return 1.0 if pred_result == actual_result else 0.0
    
    def calculate_score_accuracy(self, predictions: List[Dict], actual: Dict) -> float:
        """حساب دقة توقع النتيجة الدقيقة"""
        if not predictions:
            return 0.0
        
        best_pred = predictions[0]
        pred_home = best_pred.get('home_goals', 0)
        pred_away = best_pred.get('away_goals', 0)
        actual_home = actual.get('home_goals', 0)
        actual_away = actual.get('away_goals', 0)
        
        if pred_home == actual_home and pred_away == actual_away:
            return 1.0
        elif abs(pred_home - actual_home) <= 1 and abs(pred_away - actual_away) <= 1:
            return 0.5
        else:
            return 0.0
    
    def calculate_difference_accuracy(self, predictions: List[Dict], actual: Dict) -> float:
        """حساب دقة توقع فارق الأهداف"""
        if not predictions:
            return 0.0
        
        best_pred = predictions[0]
        pred_diff = best_pred.get('home_goals', 0) - best_pred.get('away_goals', 0)
        actual_diff = actual.get('home_goals', 0) - actual.get('away_goals', 0)
        
        if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0) or (pred_diff == 0 and actual_diff == 0):
            return 1.0
        else:
            return 0.0
    
    def calculate_confidence_penalty(self, predictions: List[Dict], actual: Dict) -> float:
        """حساب عقاب الثقة (عقاب الثقة العالية في تنبؤات خاطئة)"""
        if not predictions:
            return 0.0
        
        best_pred = predictions[0]
        confidence = best_pred.get('confidence', 0.5)
        
        # تحقق إذا كان التنبؤ خاطئاً
        pred_home = best_pred.get('home_goals', 0)
        pred_away = best_pred.get('away_goals', 0)
        actual_home = actual.get('home_goals', 0)
        actual_away = actual.get('away_goals', 0)
        
        is_correct = (pred_home == actual_home and pred_away == actual_away)
        
        if not is_correct and confidence > 0.7:
            return confidence  # عقاب للثقة العالية في تنبؤ خاطئ
        
        return 0.0
    
    def update_weights(self, reward: float, next_state: np.ndarray):
        """تحديث الأوزان بناءً على المكافأة"""
        # تحديث استراتيجية الاستكشاف
        self.exploration_rate = max(self.min_exploration, 
                                  self.exploration_rate * self.exploration_decay)
        
        # تحديث الأوزان بناءً على المكافأة
        learning_adjustment = self.learning_rate * reward
        
        for key in self.weights:
            # تعديل الأوزان التي ساهمت في المكافأة الإيجابية
            if reward > 0:
                self.weights[key] += learning_adjustment * random.uniform(0.8, 1.2)
            else:
                self.weights[key] -= learning_adjustment * random.uniform(0.8, 1.2)
            
            # التأكد من أن الأوزان ضمن الحدود المعقولة
            self.weights[key] = max(0.01, min(0.3, self.weights[key]))
        
        # تطبيع الأوزان
        self.weights = self.normalize_weights(self.weights)
        
        # حفظ أفضل الأوزان
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_weights = self.weights.copy()
            self.logger.info(f"🚀 أفضل مكافأة جديدة: {reward:.3f}")
    
    def train(self, training_data: List[Dict], episodes: int = 1000, callback: Optional[Callable] = None):
        """تدريب النموذج باستخدام التعلم المعزز"""
        self.logger.info(f"🎯 بدء التدريب على {episodes} حلقة...")
        
        for episode in range(episodes):
            episode_reward = 0
            matches_trained = 0
            episode_actions = []
            episode_states = []
            
            for match_data in training_data:
                # استخراج البيانات
                team_metrics = match_data['team_metrics']
                opponent_metrics = match_data['opponent_metrics']
                context = match_data['context']
                actual_result = match_data['actual_result']
                
                # الحصول على سمات الدولة
                state = self.get_state_features(team_metrics, opponent_metrics, context)
                episode_states.append(state.tolist())
                
                # اختيار action (تعديل الأوزان)
                action_weights = self.select_action(state)
                episode_actions.append({k: v for k, v in action_weights.items()})
                
                # توليد التنبؤات باستخدام الأوزان الجديدة
                predictions = self.generate_predictions(team_metrics, opponent_metrics, context, action_weights)
                
                # حساب المكافأة
                reward = self.calculate_reward(predictions, actual_result)
                episode_reward += reward
                matches_trained += 1
                
                # تحديث الأوزان
                next_state = self.get_state_features(team_metrics, opponent_metrics, context)
                self.update_weights(reward, next_state)
                
                # حفظ في الذاكرة
                self.remember(state, action_weights, reward, next_state)
            
            # تسجيل أداء الحلقة
            avg_reward = episode_reward / matches_trained if matches_trained > 0 else 0
            self.performance_history.append(avg_reward)
            self.weights_history.append(self.weights.copy())
            
            # استدعاء callback إذا كان موجوداً
            if callback:
                callback(episode, avg_reward, self.exploration_rate, episode_actions[:5], episode_states[:3])
            
            # حفظ بيانات الحلقة
            self._save_episode_data(episode, avg_reward, self.exploration_rate)
            
            if episode % 100 == 0:
                self.logger.info(f"📊 الحلقة {episode}: متوسط المكافأة = {avg_reward:.3f}, استكشاف = {self.exploration_rate:.3f}")
        
        self.logger.info(f"✅ اكتمل التدريب! أفضل مكافأة: {self.best_reward:.3f}")
        
        # حفظ النتائج النهائية
        self.save_training_results()
    
    def _save_episode_data(self, episode: int, reward: float, exploration_rate: float):
        """حفظ بيانات الحلقة"""
        episode_data = {
            'episode_number': episode,
            'timestamp': datetime.now().isoformat(),
            'reward_achieved': reward,
            'exploration_rate': exploration_rate,
            'current_weights': {k: round(v, 4) for k, v in self.weights.items()}
        }
        
        filename = f"{self.output_dir}/episodes/episode_{episode:03d}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
    
    def save_training_results(self):
        """حفظ نتائج التدريب النهائية"""
        results = {
            'best_weights': self.best_weights,
            'best_reward': self.best_reward,
            'final_weights': self.weights,
            'performance_history': self.performance_history,
            'training_parameters': {
                'state_dim': self.state_dim,
                'learning_rate': self.learning_rate,
                'min_exploration': self.min_exploration,
                'exploration_decay': self.exploration_decay
            },
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.output_dir}/rl_training_results.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 تم حفظ نتائج التدريب في: {filename}")
    
    def remember(self, state, action, reward, next_state):
        """حفظ التجربة في الذاكرة"""
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def generate_predictions(self, team_metrics: Dict, opponent_metrics: Dict, context: Dict, weights: Dict) -> List[Dict]:
        """توليد تنبؤات باستخدام الأوزان المحددة"""
        # حساب القوة الإجمالية باستخدام الأوزان
        team_strength = self.calculate_weighted_strength(team_metrics, weights)
        opponent_strength = self.calculate_weighted_strength(opponent_metrics, weights)
        
        # تطبيق عوامل السياق
        home_advantage = context.get('home_advantage', 1.1)
        motivation_boost = (team_metrics.get('motivation_factor', 1.0) + 
                          opponent_metrics.get('motivation_factor', 1.0)) / 2
        
        # حساب الأهداف المتوقعة
        base_goals = 1.4  # متوسط أهداف الدوري
        home_expected = team_strength * (1 / opponent_strength) * base_goals * home_advantage
        away_expected = opponent_strength * (1 / team_strength) * base_goals / home_advantage
        
        # تطبيق عوامل التحفيز
        home_expected *= team_metrics.get('motivation_factor', 1.0)
        away_expected *= opponent_metrics.get('motivation_factor', 1.0)
        
        # ضمان واقعية النتائج
        home_expected = max(0.2, min(3.5, home_expected))
        away_expected = max(0.2, min(3.0, away_expected))
        
        # توليد تنبؤات متعددة
        predictions = []
        for _ in range(3):
            home_goals = np.random.poisson(home_expected)
            away_goals = np.random.poisson(away_expected)
            
            predictions.append({
                'home_goals': int(home_goals),
                'away_goals': int(away_goals),
                'confidence': np.random.uniform(0.3, 0.8),
                'type': self.classify_prediction(home_goals, away_goals)
            })
        
        return predictions
    
    def calculate_weighted_strength(self, metrics: Dict, weights: Dict) -> float:
        """حساب القوة المرجحة باستخدام الأوزان"""
        strength = 0.0
        
        for metric_key, weight in weights.items():
            if metric_key in metrics:
                metric_value = metrics[metric_key]
                # تحويل النسب المئوية إلى قيم من 0-1 إذا لزم الأمر
                if isinstance(metric_value, (int, float)) and metric_value > 1:
                    metric_value = metric_value / 100
                strength += metric_value * weight
        
        return max(0.1, strength)
    
    def classify_prediction(self, home_goals: int, away_goals: int) -> str:
        """تصنيف نوع التنبؤ"""
        goal_diff = abs(home_goals - away_goals)
        total_goals = home_goals + away_goals
        
        if goal_diff <= 1:
            return "تنبؤ آمن"
        elif total_goals >= 5:
            return "نتيجة عالية"
        elif goal_diff >= 3:
            return "نتيجة كبيرة"
        else:
            return "تنبؤ معقول"
    
    def get_optimal_weights(self) -> Dict[str, float]:
        """الحصول على الأوزان المثلى"""
        return self.best_weights
    
    def save_weights(self, filepath: str):
        """حفظ الأوزان في ملف"""
        weights_data = {
            'weights': self.best_weights,
            'performance_history': self.performance_history,
            'best_reward': self.best_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weights_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 تم حفظ الأوزان في: {filepath}")
    
    def load_weights(self, filepath: str):
        """تحميل الأوزان من ملف"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                weights_data = json.load(f)
            
            self.best_weights = weights_data['weights']
            self.weights = self.best_weights.copy()
            self.performance_history = weights_data.get('performance_history', [])
            self.best_reward = weights_data.get('best_reward', 0)
            
            self.logger.info(f"📂 تم تحميل الأوزان من: {filepath}")
        except FileNotFoundError:
            self.logger.warning(f"⚠️  ملف الأوزان غير موجود: {filepath}")