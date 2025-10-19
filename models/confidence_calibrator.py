# models/confidence_calibrator.py - النسخة المحسنة والمكتملة

import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import json

class ConfidenceCalibrator:
    def __init__(self):
        self.calibration_data = []
        self.isotonic_model = None
        self.is_calibrated = False
        self.calibration_report = {}
        
        # إعداد logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def collect_calibration_data(self, predictions: List[Dict], actual_results: List[Dict]):
        """جمع بيانات المعايرة من التنبؤات والنتائج الفعلية"""
        try:
            if len(predictions) != len(actual_results):
                self.logger.warning(f"⚠️  عدد التنبؤات ({len(predictions)}) لا يساوي عدد النتائج ({len(actual_results)})")
                min_len = min(len(predictions), len(actual_results))
                predictions = predictions[:min_len]
                actual_results = actual_results[:min_len]
            
            for pred, actual in zip(predictions, actual_results):
                if pred and actual:  # التأكد من أن البيانات غير فارغة
                    self.calibration_data.append((pred, actual))
            
            self.logger.info(f"📊 تم جمع {len(self.calibration_data)} عينة للمعايرة")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في جمع بيانات المعايرة: {e}")
    
    def fit(self):
        """تدريب نماذج المعايرة"""
        try:
            if len(self.calibration_data) < 10:
                self.logger.warning("⚠️  بيانات معايرة غير كافية")
                return False
            
            # استخراج الثقة والنتائج الفعلية
            confidences = []
            actuals = []
            
            for pred, actual in self.calibration_data:
                confidence = pred.get('confidence', 0.5)
                is_correct = self._is_prediction_correct(pred, actual)
                
                confidences.append(confidence)
                actuals.append(1 if is_correct else 0)
            
            # تدريب نموذج Isotonic Regression للمعايرة
            self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_model.fit(confidences, actuals)
            
            # إنشاء تقرير المعايرة
            self._generate_calibration_report(confidences, actuals)
            
            self.is_calibrated = True
            self.logger.info("✅ تم تدريب نموذج معايرة الثقة")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تدريب المعايرة: {e}")
            return False
    
    def _is_prediction_correct(self, prediction: Dict, actual_result: Dict) -> bool:
        """التحقق من صحة التنبؤ"""
        try:
            if not prediction or not actual_result:
                return False
            
            pred_home = prediction.get('home_goals', 0)
            pred_away = prediction.get('away_goals', 0)
            actual_home = actual_result.get('home_goals', 0)
            actual_away = actual_result.get('away_goals', 0)
            
            # اعتبار التنبؤ صحيح إذا تطابق الفائز أو التعادل
            if (pred_home > pred_away and actual_home > actual_away) or \
               (pred_home < pred_away and actual_home < actual_away) or \
               (pred_home == pred_away and actual_home == actual_away):
                return True
            
            # أيضاً اعتبار التنبؤ صحيح إذا كان الفرق في الأهداف صغير
            goal_diff_pred = abs(pred_home - pred_away)
            goal_diff_actual = abs(actual_home - actual_away)
            
            if abs(goal_diff_pred - goal_diff_actual) <= 1:
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️  خطأ في التحقق من صحة التنبؤ: {e}")
            return False
    
    def calibrate_confidence(self, confidence: float, prediction_type: str = "general") -> float:
        """معايرة قيمة الثقة"""
        try:
            if not self.is_calibrated or self.isotonic_model is None:
                return confidence
            
            # تطبيق المعايرة
            calibrated = self.isotonic_model.predict([confidence])[0]
            return max(0.0, min(1.0, calibrated))
            
        except Exception as e:
            self.logger.warning(f"⚠️  خطأ في المعايرة، استخدام الثقة الأصلية: {e}")
            return confidence
    
    def _generate_calibration_report(self, confidences: List[float], actuals: List[int]):
        """إنشاء تقرير مفصل عن المعايرة"""
        try:
            # حساب منحنى المعايرة
            prob_true, prob_pred = calibration_curve(actuals, confidences, n_bins=10, strategy='uniform')
            
            # حساب مقاييس الأداء
            calibration_error = np.mean(np.abs(prob_true - prob_pred))
            brier_score = np.mean([(actual - pred) ** 2 for actual, pred in zip(actuals, confidences)])
            
            # حساب موثوقية الثقة
            confidence_reliability = {}
            for conf_bin in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                mask = [c >= conf_bin and c < conf_bin + 0.2 for c in confidences]
                if sum(mask) > 0:
                    bin_accuracy = np.mean([actuals[i] for i, m in enumerate(mask) if m])
                    confidence_reliability[f"{conf_bin:.1f}-{conf_bin+0.2:.1f}"] = {
                        'count': sum(mask),
                        'accuracy': bin_accuracy,
                        'avg_confidence': np.mean([confidences[i] for i, m in enumerate(mask) if m])
                    }
            
            self.calibration_report = {
                'calibration_error': float(calibration_error),
                'brier_score': float(brier_score),
                'reliability': confidence_reliability,
                'total_samples': len(confidences),
                'overall_accuracy': np.mean(actuals)
            }
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء تقرير المعايرة: {e}")
    
    def _classify_prediction_type(self, prediction: Dict) -> str:
        """تصنيف نوع التنبؤ"""
        try:
            home_goals = prediction.get('home_goals', 0)
            away_goals = prediction.get('away_goals', 0)
            
            if home_goals == away_goals:
                return "draw"
            elif abs(home_goals - away_goals) <= 1:
                return "close_win"
            elif abs(home_goals - away_goals) <= 3:
                return "comfortable_win"
            else:
                return "big_win"
                
        except Exception as e:
            self.logger.warning(f"⚠️  خطأ في تصنيف التنبؤ: {e}")
            return "unknown"
    
    def generate_calibration_report(self) -> str:
        """إنشاء تقرير نصي عن المعايرة"""
        if not self.calibration_report:
            return "📊 لا توجد بيانات معايرة متاحة"
        
        report = [
            "📈 تقرير معايرة الثقة",
            "=" * 40,
            f"• خطأ المعايرة: {self.calibration_report.get('calibration_error', 0):.3f}",
            f"• درجة برايير: {self.calibration_report.get('brier_score', 0):.3f}",
            f"• الدقة العامة: {self.calibration_report.get('overall_accuracy', 0):.1%}",
            f"• عدد العينات: {self.calibration_report.get('total_samples', 0)}",
            "",
            "📊 موثوقية الثقة:"
        ]
        
        reliability = self.calibration_report.get('reliability', {})
        for bin_range, stats in reliability.items():
            report.append(f"• {bin_range}: {stats['accuracy']:.1%} دقة ({stats['count']} عينة)")
        
        return "\n".join(report)
    
    def save_calibration_models(self, filepath: str):
        """حفظ نماذج المعايرة"""
        try:
            calibration_data = {
                'is_calibrated': self.is_calibrated,
                'calibration_report': self.calibration_report,
                'calibration_data_sample': self.calibration_data[:10] if self.calibration_data else []
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 تم حفظ نماذج المعايرة في: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في حفظ نماذج المعايرة: {e}")
    
    def load_calibration_models(self, filepath: str):
        """تحميل نماذج المعايرة"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            self.is_calibrated = calibration_data.get('is_calibrated', False)
            self.calibration_report = calibration_data.get('calibration_report', {})
            
            self.logger.info(f"📂 تم تحميل نماذج المعايرة من: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل نماذج المعايرة: {e}")