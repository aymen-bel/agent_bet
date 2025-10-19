# models/neural_predictor.py - النسخة المحسنة والمصححة

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime

class NeuralWeightPredictor:
    def __init__(self, input_dim: int = 30, hidden_layers: List[int] = [64, 32, 16]):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.model = None
        self.is_trained = False
        self.best_accuracy = 0.0
        self.training_history = None
        
        # إعداد logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # إنشاء مجلد الإخراج
        self.output_dir = "output/training/models"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._build_model()
    
    def _build_model(self):
        """بناء النموذج العصبي مع الإخراج الصحيح"""
        try:
            self.logger.info("🏗️ بناء النموذج العصبي...")
            
            model = tf.keras.Sequential()
            
            # طبقة الإدخال
            model.add(tf.keras.layers.Dense(self.hidden_layers[0], activation='relu', input_shape=(self.input_dim,)))
            model.add(tf.keras.layers.Dropout(0.2))
            
            # الطبقات المخفية
            for units in self.hidden_layers[1:]:
                model.add(tf.keras.layers.Dense(units, activation='relu'))
                model.add(tf.keras.layers.Dropout(0.2))
            
            # طبقة الإخراج - إصلاح: 3 وحدات بدلاً من 15 لتطابق شكل الهدف
            model.add(tf.keras.layers.Dense(3, activation='softmax'))
            
            # تجميع النموذج
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.CosineSimilarity()]
            )
            
            self.model = model
            self.logger.info("✅ تم بناء النموذج العصبي بنجاح")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في بناء النموذج: {e}")
            raise
    
    def _prepare_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """تحضير البيانات للتدريب"""
        try:
            X = []
            y = []
            
            for match in training_data:
                # دمج المقاييس في متجه واحد
                team_features = list(match['team_metrics'].values())
                opponent_features = list(match['opponent_metrics'].values())
                context_features = list(match['context'].values()) if isinstance(match['context'], dict) else [match['context']]
                
                features = team_features + opponent_features + context_features
                # تأكد من الطول الصحيح وإملاء بالقيم الصفرية إذا لزم الأمر
                if len(features) < self.input_dim:
                    features.extend([0.0] * (self.input_dim - len(features)))
                else:
                    features = features[:self.input_dim]
                
                X.append(features)
                
                # تحويل النتيجة إلى متجه واحد-hot
                actual_result = match['actual_result']
                if isinstance(actual_result, dict) and 'home_goals' in actual_result:
                    home_goals = actual_result['home_goals']
                    away_goals = actual_result['away_goals']
                    
                    if home_goals > away_goals:
                        result_vector = [1, 0, 0]  # فوز Home
                    elif home_goals < away_goals:
                        result_vector = [0, 1, 0]  # فوز Away
                    else:
                        result_vector = [0, 0, 1]  # تعادل
                    
                    y.append(result_vector)
                else:
                    # استخدام متجه افتراضي إذا كانت البيانات غير صحيحة
                    y.append([0.33, 0.33, 0.34])
            
            X = np.array(X)
            y = np.array(y)
            
            # تقسيم البيانات
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            self.logger.info(f"📊 بيانات التدريب: {X_train.shape[0]} عينة، التحقق: {X_val.shape[0]} عينة")
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحضير البيانات: {e}")
            raise
    
    def train(self, training_data: List[Dict], epochs: int = 100):
        try:
            # تحويل البيانات إلى تنسيق رقمي مناسب
            X = []
            y = []
            
            for match in training_data:
                try:
                    # استخراج الميزات الرقمية فقط
                    features = self._extract_numeric_features(match)
                    if features and len(features) == len(self._get_feature_names()):
                        X.append(features)
                        
                        # النتائج المستهدفة
                        actual = match.get('actual_result', {})
                        home_goals = actual.get('home_goals', 0)
                        away_goals = actual.get('away_goals', 0)
                        
                        # ترميز النتيجة (فوز/تعادل/خسارة)
                        if home_goals > away_goals:
                            y.append([1, 0, 0])  # فوز المنزل
                        elif away_goals > home_goals:
                            y.append([0, 0, 1])  # فوز الضيف
                        else:
                            y.append([0, 1, 0])  # تعادل
                except Exception as e:
                    continue
            
            if len(X) < 10:
                self.logger.error("❌ بيانات تدريب غير كافية")
                return False
            
            # تحويل إلى numpy arrays مع أنواع بيانات صحيحة
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            # التدريب
            history = self.model.fit(
                X, y, 
                epochs=epochs, 
                validation_split=0.2,
                verbose=0,
                batch_size=32
            )
            
            self.is_trained = True
            self.training_history = history.history
            self.best_accuracy = max(history.history['accuracy'])
            
            # حفظ النموذج والنتائج
            self.save_training_results()
            
            self.logger.info(f"✅ تم التدريب بدقة {self.best_accuracy:.2%}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في التدريب العصبي: {e}")
            return False

    def _extract_numeric_features(self, match_data: Dict) -> List[float]:
        """استخراج الميزات الرقمية فقط"""
        features = []
        
        # مقاييس الفريق
        team_metrics = match_data.get('team_metrics', {})
        opponent_metrics = match_data.get('opponent_metrics', {})
        
        numeric_metrics = [
            'points_per_match', 'win_rate', 'goal_difference',
            'goals_per_match', 'conceded_per_match', 'current_form',
            'home_advantage', 'defensive_efficiency'
        ]
        
        for metric in numeric_metrics:
            features.append(float(team_metrics.get(metric, 0.0)))
            features.append(float(opponent_metrics.get(metric, 0.0)))
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """الحصول على أسماء الميزات"""
        numeric_metrics = [
            'points_per_match', 'win_rate', 'goal_difference',
            'goals_per_match', 'conceded_per_match', 'current_form',
            'home_advantage', 'defensive_efficiency'
        ]
        
        feature_names = []
        for metric in numeric_metrics:
            feature_names.append(f"team_{metric}")
            feature_names.append(f"opponent_{metric}")
        
        return feature_names
    
    def save_training_results(self):
        """حفظ نتائج التدريب"""
        try:
            # حفظ النموذج
            model_path = f"{self.output_dir}/neural_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
            self.model.save(model_path, save_format='keras')
            
            # حفظ معلومات التدريب
            training_info = {
                'best_accuracy': self.best_accuracy,
                'training_history': self.training_history,
                'model_architecture': {
                    'input_dim': self.input_dim,
                    'hidden_layers': self.hidden_layers
                },
                'feature_names': self._get_feature_names(),
                'saved_at': datetime.now().isoformat()
            }
            
            info_path = model_path.replace('.keras', '_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 تم حفظ النموذج والنتائج في: {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ نتائج التدريب: {e}")
    
    def predict_weights(self, team_metrics: Dict, opponent_metrics: Dict, context: Dict) -> Dict[str, float]:
        """توقع الأوزان المثلى للمقاييس"""
        try:
            if not self.is_trained:
                self.logger.warning("⚠️  النموذج غير مدرب، استخدام الأوزان الافتراضية")
                return self.get_default_weights()
            
            # تحضير بيانات الإدخال
            team_features = list(team_metrics.values())
            opponent_features = list(opponent_metrics.values())
            context_features = list(context.values()) if isinstance(context, dict) else [context]
            
            features = team_features + opponent_features + context_features
            
            # إذا كانت الميزات ناقصة، أضف أصفار
            if len(features) < self.input_dim:
                features.extend([0.0] * (self.input_dim - len(features)))
            else:
                features = features[:self.input_dim]
            
            X = np.array([features])
            
            # التنبؤ
            predictions = self.model.predict(X, verbose=0)[0]
            
            # تحويل تنبؤات الفوز/الخسارة/التعادل إلى أوزان للمقاييس
            # نستخدم نمطاً بسيطاً: عندما يكون تنبؤ الفوز مرتفعاً، نعطي أوزاناً أعلى للمقاييس الهجومية
            win_prob, loss_prob, draw_prob = predictions
            
            # إنشاء أوزان بناءً على احتمالات النتيجة
            weights = {}
            
            # المقاييس الهجومية (ترتبط باحتمال الفوز)
            offensive_metrics = ['goals_per_match', 'shot_efficiency', 'conversion_rate', 'points_per_match']
            for metric in offensive_metrics:
                if metric in team_metrics:
                    weights[metric] = win_prob * 0.3
            
            # المقاييس الدفاعية (ترتبط باحتمال الخسارة)
            defensive_metrics = ['conceded_per_match', 'defensive_efficiency', 'goal_difference']
            for metric in defensive_metrics:
                if metric in team_metrics:
                    weights[metric] = (1 - loss_prob) * 0.2
            
            # المقاييس العامة
            general_metrics = ['win_rate', 'current_form', 'motivation_factor']
            for metric in general_metrics:
                if metric in team_metrics:
                    weights[metric] = 0.1 + (win_prob - loss_prob) * 0.1
            
            # المقاييس السياقية
            context_metrics = ['home_advantage', 'match_importance', 'historical_performance']
            for metric in context_metrics:
                if metric in context:
                    weights[metric] = 0.05 + draw_prob * 0.1
            
            # التأكد من أن جميع المقاييس لها قيم
            all_metrics = set(team_metrics.keys()) | set(context.keys())
            for metric in all_metrics:
                if metric not in weights:
                    weights[metric] = 0.05
            
            # تطبيع الأوزان
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            else:
                weights = self.get_default_weights()
            
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في التنبؤ: {e}")
            return self.get_default_weights()
    
    def get_default_weights(self) -> Dict[str, float]:
        """الحصول على الأوزان الافتراضية"""
        default_weights = {
            'points_per_match': 0.15,
            'win_rate': 0.12,
            'goal_difference': 0.10,
            'goals_per_match': 0.10,
            'conceded_per_match': 0.10,
            'shot_efficiency': 0.08,
            'conversion_rate': 0.08,
            'defensive_efficiency': 0.08,
            'current_form': 0.07,
            'motivation_factor': 0.06,
            'home_advantage': 0.03,
            'match_importance': 0.02,
            'historical_performance': 0.01
        }
        
        # تطبيع الأوزان
        total = sum(default_weights.values())
        return {k: v/total for k, v in default_weights.items()}
    
    def save_model(self, filepath: str):
        """حفظ النموذج المدرب"""
        try:
            if self.is_trained and self.model:
                # استخدام formato Keras الحديث
                if filepath.endswith('.h5'):
                    filepath = filepath.replace('.h5', '.keras')
                
                self.model.save(filepath, save_format='keras')
                self.logger.info(f"💾 تم حفظ النموذج في: {filepath}")
                
                # حفظ معلومات التدريب
                training_info = {
                    'best_accuracy': self.best_accuracy,
                    'training_history': self.training_history,
                    'saved_at': datetime.now().isoformat(),
                    'input_dim': self.input_dim,
                    'hidden_layers': self.hidden_layers
                }
                
                info_path = filepath.replace('.keras', '_info.json')
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(training_info, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ النموذج: {e}")
    
    def load_model(self, filepath: str):
        """تحميل النموذج المدرب مع معالجة مشكلة optimizer"""
        try:
            # محاولة تحميل النموذج بدون optimizer
            self.model = tf.keras.models.load_model(filepath, compile=False)
            
            # إعادة تجميع النموذج مع optimizer جديد
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.CosineSimilarity()]
            )
            
            self.is_trained = True
            
            # تحميل معلومات التدريب
            info_path = filepath.replace('.keras', '_info.json').replace('.h5', '_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    training_info = json.load(f)
                    self.best_accuracy = training_info.get('best_accuracy', 0.5)
                    self.training_history = training_info.get('training_history', {})
            
            self.logger.info(f"📂 تم تحميل النموذج من: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل النموذج: {e}")