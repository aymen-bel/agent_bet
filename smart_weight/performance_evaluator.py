# performance_evaluator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class PerformanceEvaluator:
    def __init__(self):
        self.evaluation_history = []
        self.model_comparisons = []
        
    def evaluate_model_performance(self, predictions: List[Dict], actual_results: List[Dict], 
                                 model_name: str = "SmartOptimizer") -> Dict[str, float]:
        """تقييم أداء النموذج الشامل"""
        if len(predictions) != len(actual_results):
            raise ValueError("عدد التنبؤات يجب أن يساوي عدد النتائج الفعلية")
        
        metrics = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'accuracy_metrics': {},
            'confidence_metrics': {},
            'distribution_metrics': {}
        }
        
        # حساب مقاييس الدقة
        accuracy_metrics = self.calculate_accuracy_metrics(predictions, actual_results)
        metrics['accuracy_metrics'] = accuracy_metrics
        
        # حساب مقاييس الثقة
        confidence_metrics = self.calculate_confidence_metrics(predictions, actual_results)
        metrics['confidence_metrics'] = confidence_metrics
        
        # حساب مقاييس التوزيع
        distribution_metrics = self.calculate_distribution_metrics(predictions, actual_results)
        metrics['distribution_metrics'] = distribution_metrics
        
        # النتيجة الشاملة
        overall_score = self.calculate_overall_score(metrics)
        metrics['overall_score'] = overall_score
        
        # حفظ التقييم
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def calculate_accuracy_metrics(self, predictions: List[Dict], actual_results: List[Dict]) -> Dict[str, float]:
        """حساب مقاييس الدقة المختلفة"""
        total_predictions = len(predictions)
        
        # دقة توقع النتيجة (H/D/A)
        result_accuracy = self.calculate_result_accuracy(predictions, actual_results)
        
        # دقة توقع النتيجة الدقيقة
        exact_score_accuracy = self.calculate_exact_score_accuracy(predictions, actual_results)
        
        # دقة توقع الفارق
        goal_difference_accuracy = self.calculate_goal_difference_accuracy(predictions, actual_results)
        
        # دقة توقع أكثر/أقل من 2.5 هدف
        over_under_accuracy = self.calculate_over_under_accuracy(predictions, actual_results)
        
        # دقة توقع تسجيل الفريقين
        both_teams_score_accuracy = self.calculate_both_teams_score_accuracy(predictions, actual_results)
        
        return {
            'result_accuracy': result_accuracy,
            'exact_score_accuracy': exact_score_accuracy,
            'goal_difference_accuracy': goal_difference_accuracy,
            'over_under_accuracy': over_under_accuracy,
            'both_teams_score_accuracy': both_teams_score_accuracy,
            'weighted_accuracy': self.calculate_weighted_accuracy(
                result_accuracy, exact_score_accuracy, goal_difference_accuracy
            )
        }
    
    def calculate_result_accuracy(self, predictions: List[Dict], actual_results: List[Dict]) -> float:
        """حساب دقة توقع النتيجة (فوز/تعادل/خسارة)"""
        correct_predictions = 0
        
        for pred, actual in zip(predictions, actual_results):
            pred_home = pred['home_goals']
            pred_away = pred['away_goals']
            actual_home = actual['home_goals']
            actual_away = actual['away_goals']
            
            pred_result = 'H' if pred_home > pred_away else 'A' if pred_away > pred_home else 'D'
            actual_result = 'H' if actual_home > actual_away else 'A' if actual_away > actual_home else 'D'
            
            if pred_result == actual_result:
                correct_predictions += 1
        
        return correct_predictions / len(predictions) if predictions else 0.0
    
    def calculate_exact_score_accuracy(self, predictions: List[Dict], actual_results: List[Dict]) -> float:
        """حساب دقة توقع النتيجة الدقيقة"""
        correct_predictions = 0
        
        for pred, actual in zip(predictions, actual_results):
            if (pred['home_goals'] == actual['home_goals'] and 
                pred['away_goals'] == actual['away_goals']):
                correct_predictions += 1
        
        return correct_predictions / len(predictions) if predictions else 0.0
    
    def calculate_goal_difference_accuracy(self, predictions: List[Dict], actual_results: List[Dict]) -> float:
        """حساب دقة توقع فارق الأهداف"""
        correct_predictions = 0
        
        for pred, actual in zip(predictions, actual_results):
            pred_diff = pred['home_goals'] - pred['away_goals']
            actual_diff = actual['home_goals'] - actual['away_goals']
            
            if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0) or (pred_diff == 0 and actual_diff == 0):
                correct_predictions += 1
        
        return correct_predictions / len(predictions) if predictions else 0.0
    
    def calculate_over_under_accuracy(self, predictions: List[Dict], actual_results: List[Dict]) -> float:
        """حساب دقة توقع أكثر/أقل من 2.5 هدف"""
        correct_predictions = 0
        
        for pred, actual in zip(predictions, actual_results):
            pred_total = pred['home_goals'] + pred['away_goals']
            actual_total = actual['home_goals'] + actual['away_goals']
            
            pred_over = pred_total > 2.5
            actual_over = actual_total > 2.5
            
            if pred_over == actual_over:
                correct_predictions += 1
        
        return correct_predictions / len(predictions) if predictions else 0.0
    
    def calculate_both_teams_score_accuracy(self, predictions: List[Dict], actual_results: List[Dict]) -> float:
        """حساب دقة توقع تسجيل الفريقين"""
        correct_predictions = 0
        
        for pred, actual in zip(predictions, actual_results):
            pred_both = pred['home_goals'] > 0 and pred['away_goals'] > 0
            actual_both = actual['home_goals'] > 0 and actual['away_goals'] > 0
            
            if pred_both == actual_both:
                correct_predictions += 1
        
        return correct_predictions / len(predictions) if predictions else 0.0
    
    def calculate_weighted_accuracy(self, result_acc: float, exact_acc: float, diff_acc: float) -> float:
        """حساب الدقة المرجحة"""
        return (result_acc * 0.5 + exact_acc * 0.2 + diff_acc * 0.3)
    
    def calculate_confidence_metrics(self, predictions: List[Dict], actual_results: List[Dict]) -> Dict[str, float]:
        """حساب مقاييس الثقة"""
        confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        for pred, actual in zip(predictions, actual_results):
            confidence = pred.get('confidence', 0.5)
            confidences.append(confidence)
            
            # التحقق من صحة التنبؤ
            pred_home = pred['home_goals']
            pred_away = pred['away_goals']
            actual_home = actual['home_goals']
            actual_away = actual['away_goals']
            
            is_correct = (pred_home == actual_home and pred_away == actual_away)
            
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
        
        # معايرة الثقة
        calibration_error = self.calculate_calibration_error(confidences, predictions, actual_results)
        
        return {
            'average_confidence': np.mean(confidences) if confidences else 0.5,
            'correct_confidence': np.mean(correct_confidences) if correct_confidences else 0.5,
            'incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0.5,
            'calibration_error': calibration_error,
            'confidence_spread': np.std(confidences) if confidences else 0.1
        }
    
    def calculate_calibration_error(self, confidences: List[float], predictions: List[Dict], actual_results: List[Dict]) -> float:
        """حساب خطأ معايرة الثقة"""
        if not confidences:
            return 1.0
        
        # تجميع الثقة في فئات
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        calibration_error = 0
        total_weight = 0
        
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            bin_indices = [j for j, conf in enumerate(confidences) if lower <= conf < upper]
            
            if bin_indices:
                bin_confidence = np.mean([confidences[j] for j in bin_indices])
                bin_accuracy = self.calculate_bin_accuracy(bin_indices, predictions, actual_results)
                
                calibration_error += abs(bin_confidence - bin_accuracy) * len(bin_indices)
                total_weight += len(bin_indices)
        
        return calibration_error / total_weight if total_weight > 0 else 1.0
    
    def calculate_bin_accuracy(self, bin_indices: List[int], predictions: List[Dict], actual_results: List[Dict]) -> float:
        """حساب دقة تنبؤات فئة ثقة محددة"""
        correct = 0
        total = len(bin_indices)
        
        for idx in bin_indices:
            pred = predictions[idx]
            actual = actual_results[idx]
            
            if (pred['home_goals'] == actual['home_goals'] and 
                pred['away_goals'] == actual['away_goals']):
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def calculate_distribution_metrics(self, predictions: List[Dict], actual_results: List[Dict]) -> Dict[str, float]:
        """حساب مقاييس توزيع التنبؤات"""
        pred_home_goals = [p['home_goals'] for p in predictions]
        pred_away_goals = [p['away_goals'] for p in predictions]
        actual_home_goals = [a['home_goals'] for a in actual_results]
        actual_away_goals = [a['away_goals'] for a in actual_results]
        
        return {
            'pred_home_mean': np.mean(pred_home_goals) if pred_home_goals else 0,
            'pred_away_mean': np.mean(pred_away_goals) if pred_away_goals else 0,
            'actual_home_mean': np.mean(actual_home_goals) if actual_home_goals else 0,
            'actual_away_mean': np.mean(actual_away_goals) if actual_away_goals else 0,
            'home_goal_mse': np.mean((np.array(pred_home_goals) - np.array(actual_home_goals))**2) if pred_home_goals else 0,
            'away_goal_mse': np.mean((np.array(pred_away_goals) - np.array(actual_away_goals))**2) if pred_away_goals else 0,
            'total_goal_correlation': np.corrcoef(pred_home_goals + pred_away_goals, 
                                                actual_home_goals + actual_away_goals)[0, 1] if len(pred_home_goals) > 1 else 0
        }
    
    def calculate_overall_score(self, metrics: Dict) -> float:
        """حساب النتيجة الشاملة للنموذج"""
        accuracy_metrics = metrics['accuracy_metrics']
        confidence_metrics = metrics['confidence_metrics']
        distribution_metrics = metrics['distribution_metrics']
        
        # أوزان المكونات المختلفة
        weights = {
            'accuracy': 0.6,
            'confidence': 0.25,
            'distribution': 0.15
        }
        
        # حساب درجة الدقة
        accuracy_score = (
            accuracy_metrics['weighted_accuracy'] * 0.7 +
            accuracy_metrics['over_under_accuracy'] * 0.2 +
            accuracy_metrics['both_teams_score_accuracy'] * 0.1
        )
        
        # حساب درجة الثقة
        confidence_score = max(0, 1 - confidence_metrics['calibration_error'])
        
        # حساب درجة التوزيع
        distribution_score = max(0, 1 - (distribution_metrics['home_goal_mse'] + distribution_metrics['away_goal_mse']) / 2)
        
        # النتيجة الشاملة
        overall_score = (
            accuracy_score * weights['accuracy'] +
            confidence_score * weights['confidence'] +
            distribution_score * weights['distribution']
        )
        
        return overall_score
    
    def compare_models(self, model_results: List[Dict]) -> pd.DataFrame:
        """مقارنة أداء نماذج متعددة"""
        comparison_data = []
        
        for result in model_results:
            comparison_data.append({
                'model_name': result['model_name'],
                'overall_score': result['overall_score'],
                'result_accuracy': result['accuracy_metrics']['result_accuracy'],
                'exact_accuracy': result['accuracy_metrics']['exact_score_accuracy'],
                'weighted_accuracy': result['accuracy_metrics']['weighted_accuracy'],
                'calibration_error': result['confidence_metrics']['calibration_error'],
                'total_predictions': result['total_predictions']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('overall_score', ascending=False)
        
        self.model_comparisons.append({
            'timestamp': datetime.now().isoformat(),
            'comparison': comparison_df.to_dict('records')
        })
        
        return comparison_df
    
    def generate_performance_report(self, metrics: Dict, save_path: str = None) -> str:
        """إنشاء تقرير أداء مفصل"""
        report = {
            'title': 'تقرير أداء النموذج الذكي',
            'timestamp': metrics['timestamp'],
            'summary': self.generate_summary(metrics),
            'detailed_metrics': metrics,
            'recommendations': self.generate_recommendations(metrics)
        }
        
        report_text = f"""
📊 تقرير أداء النموذج الذكي
{'='*50}

🎯 الملخص:
{report['summary']}

📈 المقاييس التفصيلية:
• الدقة الشاملة: {metrics['overall_score']:.3f}
• دقة توقع النتيجة: {metrics['accuracy_metrics']['result_accuracy']:.3f}
• دقة توقع النتيجة الدقيقة: {metrics['accuracy_metrics']['exact_score_accuracy']:.3f}
• دقة توقع فارق الأهداف: {metrics['accuracy_metrics']['goal_difference_accuracy']:.3f}
• خطأ معايرة الثقة: {metrics['confidence_metrics']['calibration_error']:.3f}

💡 التوصيات:
{report['recommendations']}
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ التقرير في: {save_path}")
        
        return report_text
    
    def generate_summary(self, metrics: Dict) -> str:
        """توليد ملخص الأداء"""
        overall_score = metrics['overall_score']
        
        if overall_score >= 0.7:
            rating = "ممتاز"
            color = "🟢"
        elif overall_score >= 0.6:
            rating = "جيد جداً"
            color = "🟡"
        elif overall_score >= 0.5:
            rating = "مقبول"
            color = "🟠"
        else:
            rating = "بحاجة للتحسين"
            color = "🔴"
        
        return f"{color} التقييم: {rating} ({(overall_score*100):.1f}%)"
    
    def generate_recommendations(self, metrics: Dict) -> str:
        """توليد توصيات للتحسين"""
        recommendations = []
        
        if metrics['accuracy_metrics']['result_accuracy'] < 0.6:
            recommendations.append("• تحسين دقة توقع النتيجة الأساسية")
        
        if metrics['confidence_metrics']['calibration_error'] > 0.2:
            recommendations.append("• معايرة أفضل لنظام الثقة")
        
        if metrics['accuracy_metrics']['exact_score_accuracy'] < 0.1:
            recommendations.append("• تحسين توقع النتائج الدقيقة")
        
        if not recommendations:
            recommendations.append("• الاستمرار في التدريب مع بيانات جديدة")
            recommendations.append("• اختبار النموذج على فترات زمنية مختلفة")
        
        return "\n".join(recommendations)
    
    def plot_performance_trends(self, save_path: str = None):
        """رسم trends أداء النموذج عبر الزمن"""
        if len(self.evaluation_history) < 2:
            print("⚠️  لا توجد بيانات كافية لرسم الاتجاهات")
            return
        
        # إعداد الرسم
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('اتجاهات أداء النموذج الذكي', fontsize=16)
        
        # استخراج البيانات
        timestamps = [i for i in range(len(self.evaluation_history))]
        overall_scores = [e['overall_score'] for e in self.evaluation_history]
        result_accuracies = [e['accuracy_metrics']['result_accuracy'] for e in self.evaluation_history]
        exact_accuracies = [e['accuracy_metrics']['exact_score_accuracy'] for e in self.evaluation_history]
        calibration_errors = [e['confidence_metrics']['calibration_error'] for e in self.evaluation_history]
        
        # الرسم 1: النتيجة الشاملة
        axes[0, 0].plot(timestamps, overall_scores, 'b-', linewidth=2, marker='o')
        axes[0, 0].set_title('النتيجة الشاملة')
        axes[0, 0].set_ylabel('النسبة')
        axes[0, 0].grid(True, alpha=0.3)
        
        # الرسم 2: دقة النتائج
        axes[0, 1].plot(timestamps, result_accuracies, 'g-', linewidth=2, marker='s', label='نتيجة')
        axes[0, 1].plot(timestamps, exact_accuracies, 'r-', linewidth=2, marker='^', label='دقيقة')
        axes[0, 1].set_title('دقة التنبؤات')
        axes[0, 1].set_ylabel('النسبة')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # الرسم 3: خطأ المعايرة
        axes[1, 0].plot(timestamps, calibration_errors, 'm-', linewidth=2, marker='d')
        axes[1, 0].set_title('خطأ معايرة الثقة')
        axes[1, 0].set_ylabel('الخطأ')
        axes[1, 0].set_xlabel('التقييمات')
        axes[1, 0].grid(True, alpha=0.3)
        
        # الرسم 4: histogram النتائج الشاملة
        axes[1, 1].hist(overall_scores, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('توزيع النتائج الشاملة')
        axes[1, 1].set_xlabel('النتيجة')
        axes[1, 1].set_ylabel('التكرار')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ تم حفظ الرسم في: {save_path}")
        
        plt.show()