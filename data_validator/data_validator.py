# data_validator.py
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DataValidator:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.predictions_history = {}
        self.performance_metrics = {}
        self.validation_results = {}
        
    def validate_and_clean_data(self):
        """التحقق من جودة البيانات وإصلاحها"""
        print("🔍 التحقق من جودة البيانات...")
        
        # التحقق من الأعمدة الأساسية
        required_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Date']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            print(f"❌ أعمدة مفقودة: {missing_columns}")
            return False
        
        # تنظيف أسماء الفرق
        self.data['HomeTeam'] = self.data['HomeTeam'].astype(str).str.strip().str.title()
        self.data['AwayTeam'] = self.data['AwayTeam'].astype(str).str.strip().str.title()
        
        # إزالة المباريات غير الصالحة
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        self.data = self.data[self.data['HomeTeam'] != self.data['AwayTeam']]
        
        # تنظيف البيانات الرقمية
        self._clean_numeric_data()
        
        print(f"✅ تم تنظيف البيانات: {initial_count} → {len(self.data)} مباراة")
        
        # تحليل جودة البيانات
        self._analyze_data_quality()
        
        return True
    
    def _clean_numeric_data(self):
        """تنظيف البيانات الرقمية"""
        numeric_columns = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
        
        for col in numeric_columns:
            if col in self.data.columns:
                # استبدال القيم غير الصالحة بصفر
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0).astype(int)
                # ضمان أن القيم غير سالبة
                self.data[col] = self.data[col].clip(lower=0)
    
    def _analyze_data_quality(self):
        """تحليل جودة البيانات"""
        print("\n📊 تحليل جودة البيانات:")
        print(f"• إجمالي المباريات: {len(self.data)}")
        print(f"• عدد الفرق: {len(set(self.data['HomeTeam']) | set(self.data['AwayTeam']))}")
        
        # تحليل التواريخ
        if 'Date' in self.data.columns:
            try:
                self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
                valid_dates = self.data['Date'].notna()
                date_range = self.data[valid_dates]['Date']
                if len(date_range) > 0:
                    print(f"• نطاق التواريخ: {date_range.min().strftime('%Y-%m-%d')} إلى {date_range.max().strftime('%Y-%m-%d')}")
            except:
                print("• نطاق التواريخ: غير متوفر")
        
        # توزيع النتائج
        result_dist = self.data['FTR'].value_counts()
        print(f"• توزيع النتائج: H={result_dist.get('H', 0)}, D={result_dist.get('D', 0)}, A={result_dist.get('A', 0)}")
        
        # متوسط الأهداف
        avg_goals = (self.data['FTHG'].mean() + self.data['FTAG'].mean())
        print(f"• متوسط الأهداف: {avg_goals:.2f} لكل مباراة")
        
        # تحليل الاكتمال
        self._analyze_data_completeness()
    
    def _analyze_data_completeness(self):
        """تحليل اكتمال البيانات"""
        print(f"\n📋 تحليل اكتمال البيانات:")
        total_matches = len(self.data)
        
        # التحقق من وجود البيانات الإضافية
        additional_stats = ['HS', 'AS', 'HST', 'AST']
        available_stats = [stat for stat in additional_stats if stat in self.data.columns]
        
        print(f"• الإحصائيات الإضافية المتاحة: {len(available_stats)} من {len(additional_stats)}")
        
        for stat in available_stats:
            completeness = (self.data[stat].notna().sum() / total_matches) * 100
            print(f"  - {stat}: {completeness:.1f}% مكتمل")
    
    def record_prediction(self, prediction_id, prediction_data, match_info):
        """تسجيل التنبؤات للمقارنة لاحقاً"""
        self.predictions_history[prediction_id] = {
            'prediction': prediction_data,
            'match_info': match_info,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluated': False
        }
        print(f"✅ تم تسجيل التنبؤ: {prediction_id}")
    
    def validate_predictions_accuracy(self, actual_results_file=None, use_internal_data=True):
        """التحقق من دقة التنبؤات مقابل النتائج الفعلية"""
        print("\n🎯 التحقق من دقة التنبؤات...")
        
        if not self.predictions_history:
            print("❌ لا توجد تنبؤات مسجلة للتحقق")
            return None
        
        evaluation_results = {}
        total_predictions = len(self.predictions_history)
        evaluated_count = 0
        
        for pred_id, pred_data in self.predictions_history.items():
            match_info = pred_data['match_info']
            home_team = match_info['home_team']
            away_team = match_info['away_team']
            match_date = match_info.get('date')
            
            # البحث عن النتيجة الفعلية
            actual_result = self._find_actual_result(home_team, away_team, match_date, actual_results_file, use_internal_data)
            
            if actual_result:
                accuracy_metrics = self._calculate_prediction_accuracy(pred_data['prediction'], actual_result)
                evaluation_results[pred_id] = {
                    'prediction': pred_data['prediction'],
                    'actual_result': actual_result,
                    'accuracy_metrics': accuracy_metrics,
                    'match_info': match_info
                }
                self.predictions_history[pred_id]['evaluated'] = True
                evaluated_count += 1
        
        print(f"✅ تم تقييم {evaluated_count} من أصل {total_predictions} تنبؤ")
        
        if evaluated_count > 0:
            overall_metrics = self._calculate_overall_accuracy(evaluation_results)
            self.performance_metrics = overall_metrics
            self.validation_results = evaluation_results
            
            self._generate_accuracy_report(overall_metrics, evaluation_results)
            return overall_metrics
        else:
            print("❌ لم يتم العثور على نتائج فعلية لأي تنبؤ")
            return None
    
    def _find_actual_result(self, home_team, away_team, match_date, actual_results_file, use_internal_data):
        """البحث عن النتيجة الفعلية للمباراة"""
        if use_internal_data and not actual_results_file:
            # البحث في البيانات الداخلية
            match_condition = (
                (self.data['HomeTeam'].str.contains(home_team, case=False, na=False)) |
                (self.data['HomeTeam'] == home_team)
            ) & (
                (self.data['AwayTeam'].str.contains(away_team, case=False, na=False)) |
                (self.data['AwayTeam'] == away_team)
            )
            
            if match_date:
                try:
                    match_date = pd.to_datetime(match_date)
                    match_condition &= (self.data['Date'] == match_date)
                except:
                    pass
            
            matches = self.data[match_condition]
            
            if len(matches) > 0:
                match = matches.iloc[0]
                return {
                    'home_goals': match['FTHG'],
                    'away_goals': match['FTAG'],
                    'result': match['FTR'],
                    'home_team': match['HomeTeam'],
                    'away_team': match['AwayTeam'],
                    'date': match['Date'] if 'Date' in match else None
                }
        
        elif actual_results_file:
            # تحميل البيانات من ملف خارجي
            try:
                actual_data = pd.read_csv(actual_results_file)
                # تطبيق نفس منطق البحث (يمكن تخصيصه حسب هيكل الملف)
                match_condition = (
                    (actual_data['HomeTeam'] == home_team) &
                    (actual_data['AwayTeam'] == away_team)
                )
                matches = actual_data[match_condition]
                
                if len(matches) > 0:
                    match = matches.iloc[0]
                    return {
                        'home_goals': match['FTHG'],
                        'away_goals': match['FTAG'],
                        'result': match['FTR'],
                        'home_team': match['HomeTeam'],
                        'away_team': match['AwayTeam'],
                        'date': match['Date'] if 'Date' in match else None
                    }
            except Exception as e:
                print(f"❌ خطأ في تحميل ملف النتائج: {e}")
        
        return None
    
    def _calculate_prediction_accuracy(self, prediction, actual_result):
        """حساب دقة التنبؤ الفردي"""
        metrics = {}
        
        # دقة نتيجة المباراة (H/D/A)
        pred_result = self._determine_predicted_result(prediction)
        actual_FTR = actual_result['result']
        
        metrics['result_accuracy'] = 1 if pred_result == actual_FTR else 0
        metrics['predicted_result'] = pred_result
        metrics['actual_result'] = actual_FTR
        
        # دقة توقع الفوز/التعادل/الخسارة
        metrics['outcome_accuracy'] = self._calculate_outcome_accuracy(prediction, actual_result)
        
        # دقة توقع الأهداف
        metrics['score_accuracy'] = self._calculate_score_accuracy(prediction, actual_result)
        
        # دقة توقع الفارق
        metrics['goal_difference_accuracy'] = self._calculate_goal_difference_accuracy(prediction, actual_result)
        
        # دقة توقع أكثر من 2.5 هدف
        metrics['over_under_accuracy'] = self._calculate_over_under_accuracy(prediction, actual_result)
        
        # دقة توقع تسجيل الفريقين
        metrics['both_teams_score_accuracy'] = self._calculate_both_teams_score_accuracy(prediction, actual_result)
        
        return metrics
    
    def _determine_predicted_result(self, prediction):
        """تحديد النتيجة المتوقعة من بيانات التنبؤ"""
        if 'multiple_predictions' in prediction:
            # استخدام أفضل تنبؤ
            best_pred = prediction['multiple_predictions'][0]
            home_goals = best_pred['home_goals']
            away_goals = best_pred['away_goals']
        elif 'score_prediction' in prediction:
            home_goals = prediction['score_prediction']['home_goals']
            away_goals = prediction['score_prediction']['away_goals']
        else:
            # استخدام الاحتمالات
            probs = prediction.get('probabilities', {})
            if probs:
                max_prob = max(probs.get('home_win', 0), probs.get('draw', 0), probs.get('away_win', 0))
                if max_prob == probs.get('home_win', 0):
                    return 'H'
                elif max_prob == probs.get('away_win', 0):
                    return 'A'
                else:
                    return 'D'
            return 'D'  # افتراضي
        
        if home_goals > away_goals:
            return 'H'
        elif away_goals > home_goals:
            return 'A'
        else:
            return 'D'
    
    def _calculate_outcome_accuracy(self, prediction, actual_result):
        """حساب دقة توقع النتيجة (فوز/تعادل/خسارة)"""
        pred_result = self._determine_predicted_result(prediction)
        actual_FTR = actual_result['result']
        return 1 if pred_result == actual_FTR else 0
    
    def _calculate_score_accuracy(self, prediction, actual_result):
        """حساب دقة توقع النتيجة الدقيقة"""
        if 'multiple_predictions' in prediction:
            best_pred = prediction['multiple_predictions'][0]
            pred_home = best_pred['home_goals']
            pred_away = best_pred['away_goals']
        elif 'score_prediction' in prediction:
            pred_home = prediction['score_prediction']['home_goals']
            pred_away = prediction['score_prediction']['away_goals']
        else:
            return 0  # لا توجد توقعات للأهداف
        
        actual_home = actual_result['home_goals']
        actual_away = actual_result['away_goals']
        
        return 1 if (pred_home == actual_home and pred_away == actual_away) else 0
    
    def _calculate_goal_difference_accuracy(self, prediction, actual_result):
        """حساب دقة توقع فارق الأهداف"""
        if 'multiple_predictions' in prediction:
            best_pred = prediction['multiple_predictions'][0]
            pred_diff = best_pred['home_goals'] - best_pred['away_goals']
        elif 'score_prediction' in prediction:
            pred_diff = prediction['score_prediction']['home_goals'] - prediction['score_prediction']['away_goals']
        else:
            return 0
        
        actual_diff = actual_result['home_goals'] - actual_result['away_goals']
        
        # تعتبر صحيحة إذا كان الفارق في نفس الاتجاه
        if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0) or (pred_diff == 0 and actual_diff == 0):
            return 1
        return 0
    
    def _calculate_over_under_accuracy(self, prediction, actual_result):
        """حساب دقة توقع أكثر/أقل من 2.5 هدف"""
        total_goals = actual_result['home_goals'] + actual_result['away_goals']
        actual_over = total_goals > 2.5
        
        pred_probs = prediction.get('probabilities', {})
        pred_over_prob = pred_probs.get('over_2_5', 0.5)
        pred_over = pred_over_prob > 0.5
        
        return 1 if pred_over == actual_over else 0
    
    def _calculate_both_teams_score_accuracy(self, prediction, actual_result):
        """حساب دقة توقع تسجيل الفريقين"""
        actual_both_score = actual_result['home_goals'] > 0 and actual_result['away_goals'] > 0
        
        pred_probs = prediction.get('probabilities', {})
        pred_both_prob = pred_probs.get('both_teams_score', 0.5)
        pred_both_score = pred_both_prob > 0.5
        
        return 1 if pred_both_score == actual_both_score else 0
    
    def _calculate_overall_accuracy(self, evaluation_results):
        """حساب الدقة الشاملة"""
        total_predictions = len(evaluation_results)
        
        # تجميع جميع المقاييس
        result_accuracy = []
        outcome_accuracy = []
        score_accuracy = []
        goal_diff_accuracy = []
        over_under_accuracy = []
        both_teams_accuracy = []
        
        for pred_id, data in evaluation_results.items():
            metrics = data['accuracy_metrics']
            result_accuracy.append(metrics['result_accuracy'])
            outcome_accuracy.append(metrics['outcome_accuracy'])
            score_accuracy.append(metrics['score_accuracy'])
            goal_diff_accuracy.append(metrics['goal_difference_accuracy'])
            over_under_accuracy.append(metrics['over_under_accuracy'])
            both_teams_accuracy.append(metrics['both_teams_score_accuracy'])
        
        overall_metrics = {
            'total_predictions': total_predictions,
            'result_accuracy': np.mean(result_accuracy),
            'outcome_accuracy': np.mean(outcome_accuracy),
            'score_accuracy': np.mean(score_accuracy),
            'goal_difference_accuracy': np.mean(goal_diff_accuracy),
            'over_under_accuracy': np.mean(over_under_accuracy),
            'both_teams_score_accuracy': np.mean(both_teams_accuracy),
            'overall_accuracy': np.mean([
                np.mean(result_accuracy),
                np.mean(outcome_accuracy),
                np.mean(goal_diff_accuracy)
            ])
        }
        
        return overall_metrics
    
    def _generate_accuracy_report(self, overall_metrics, evaluation_results):
        """إنشاء تقرير مفصل عن الدقة"""
        print("\n" + "="*80)
        print("📊 تقرير دقة التنبؤات الشامل")
        print("="*80)
        
        print(f"\n📈 الدقة الشاملة: {overall_metrics['overall_accuracy']:.1%}")
        print(f"• إجمالي التنبؤات المُقيمة: {overall_metrics['total_predictions']}")
        
        print(f"\n🎯 تفصيل الدقة:")
        print(f"• دقة توقع النتيجة (H/D/A): {overall_metrics['result_accuracy']:.1%}")
        print(f"• دقة توقع الفوز/التعادل/الخسارة: {overall_metrics['outcome_accuracy']:.1%}")
        print(f"• دقة توقع النتيجة الدقيقة: {overall_metrics['score_accuracy']:.1%}")
        print(f"• دقة توقع فارق الأهداف: {overall_metrics['goal_difference_accuracy']:.1%}")
        print(f"• دقة توقع أكثر/أقل من 2.5 هدف: {overall_metrics['over_under_accuracy']:.1%}")
        print(f"• دقة توقع تسجيل الفريقين: {overall_metrics['both_teams_score_accuracy']:.1%}")
        
        # تحليل حسب نوع النتيجة
        self._analyze_by_result_type(evaluation_results)
        
        # أفضل وأسوأ التنبؤات
        self._show_extreme_predictions(evaluation_results)
    
    def _analyze_by_result_type(self, evaluation_results):
        """تحليل الدقة حسب نوع النتيجة"""
        result_types = {'H': [], 'D': [], 'A': []}
        
        for pred_id, data in evaluation_results.items():
            actual_result = data['actual_result']['result']
            accuracy = data['accuracy_metrics']['result_accuracy']
            result_types[actual_result].append(accuracy)
        
        print(f"\n📊 تحليل الدقة حسب نوع النتيجة:")
        for result_type, accuracies in result_types.items():
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                count = len(accuracies)
                result_name = {'H': 'فوز المنزل', 'D': 'تعادل', 'A': 'فوز الضيف'}[result_type]
                print(f"• {result_name}: {avg_accuracy:.1%} ({count} مباراة)")
    
    def _show_extreme_predictions(self, evaluation_results):
        """عرض أفضل وأسوأ التنبؤات"""
        predictions_with_accuracy = []
        
        for pred_id, data in evaluation_results.items():
            accuracy = data['accuracy_metrics']['result_accuracy']
            match_info = data['match_info']
            prediction = data['prediction']
            actual = data['actual_result']
            
            predictions_with_accuracy.append({
                'pred_id': pred_id,
                'accuracy': accuracy,
                'home_team': match_info['home_team'],
                'away_team': match_info['away_team'],
                'predicted_result': data['accuracy_metrics']['predicted_result'],
                'actual_result': data['accuracy_metrics']['actual_result'],
                'predicted_score': self._get_predicted_score(prediction),
                'actual_score': f"{actual['home_goals']}-{actual['away_goals']}"
            })
        
        # أفضل التنبؤات
        best_predictions = sorted(predictions_with_accuracy, key=lambda x: x['accuracy'], reverse=True)[:3]
        worst_predictions = sorted(predictions_with_accuracy, key=lambda x: x['accuracy'])[:3]
        
        print(f"\n🏆 أفضل 3 تنبؤات:")
        for i, pred in enumerate(best_predictions, 1):
            print(f"  {i}. {pred['home_team']} vs {pred['away_team']}: "
                  f"متوقع {pred['predicted_result']} ({pred['predicted_score']}) | "
                  f"فعلي {pred['actual_result']} ({pred['actual_score']}) | "
                  f"دقة: {pred['accuracy']:.0%}")
        
        print(f"\n📉 أسوأ 3 تنبؤات:")
        for i, pred in enumerate(worst_predictions, 1):
            print(f"  {i}. {pred['home_team']} vs {pred['away_team']}: "
                  f"متوقع {pred['predicted_result']} ({pred['predicted_score']}) | "
                  f"فعلي {pred['actual_result']} ({pred['actual_score']}) | "
                  f"دقة: {pred['accuracy']:.0%}")
    
    def _get_predicted_score(self, prediction):
        """الحصول على النتيجة المتوقعة كسلسلة"""
        if 'multiple_predictions' in prediction:
            best_pred = prediction['multiple_predictions'][0]
            return f"{best_pred['home_goals']}-{best_pred['away_goals']}"
        elif 'score_prediction' in prediction:
            return f"{prediction['score_prediction']['home_goals']}-{prediction['score_prediction']['away_goals']}"
        else:
            return "غير متوفر"
    
    def evaluate_model_performance(self, time_period=None):
        """تقييم أداء النموذج مع الزمن"""
        print("\n📈 تقييم أداء النموذج...")
        
        if not self.validation_results:
            print("❌ لا توجد نتائج تحقق لتقييم الأداء")
            return
        
        # تحليل الأداء مع الزمن (إذا كانت التواريخ متاحة)
        dated_predictions = []
        for pred_id, data in self.validation_results.items():
            match_info = data['match_info']
            if 'date' in match_info and match_info['date']:
                try:
                    date = pd.to_datetime(match_info['date'])
                    accuracy = data['accuracy_metrics']['result_accuracy']
                    dated_predictions.append({'date': date, 'accuracy': accuracy})
                except:
                    continue
        
        if dated_predictions:
            dated_df = pd.DataFrame(dated_predictions)
            dated_df = dated_df.sort_values('date')
            
            # حساب متوسط متحرك للدقة
            dated_df['moving_avg'] = dated_df['accuracy'].rolling(window=5, min_periods=1).mean()
            
            print(f"\n📅 تحليل الأداء الزمني:")
            print(f"• أول تنبؤ: {dated_df['date'].min().strftime('%Y-%m-%d')}")
            print(f"• آخر تنبؤ: {dated_df['date'].max().strftime('%Y-%m-%d')}")
            print(f"• متوسط الدقة الزمني: {dated_df['accuracy'].mean():.1%}")
            print(f"• اتجاه الأداء: {'تحسن' if dated_df['moving_avg'].iloc[-1] > dated_df['moving_avg'].iloc[0] else 'تراجع'}")
        
        # تحليل الأداء حسب قوة الفرق
        self._analyze_performance_by_team_strength()
    
    def _analyze_performance_by_team_strength(self):
        """تحليل الأداء حسب قوة الفرق"""
        print(f"\n🏅 تحليل الأداء حسب قوة الفرق:")
        
        # هذا يتطلب بيانات عن قوة الفرق - يمكن توسيعه لاحقاً
        strength_categories = {
            'مباريات كبيرة': ['Man City', 'Liverpool', 'Arsenal', 'Chelsea', 'Man United', 'Tottenham'],
            'فرق متوسطة': ['Newcastle', 'West Ham', 'Brighton', 'Aston Villa', 'Crystal Palace'],
            'فرق صغيرة': ['Bournemouth', 'Brentford', 'Fulham', 'Wolves', 'Everton']
        }
        
        # تحليل مبسط - يمكن تحسينه بمزيد من البيانات
        print("• ملاحظة: تحليل قوة الفرق يتطلب بيانات إضافية عن تصنيف الفرق")
    
    def generate_performance_report(self, save_to_file=False):
        """إنشاء تقرير أداء شامل"""
        if not self.performance_metrics:
            print("❌ لا توجد مقاييس أداء متاحة")
            return
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance_metrics': self.performance_metrics,
            'total_evaluated_predictions': len(self.validation_results),
            'summary': self._generate_performance_summary()
        }
        
        print("\n" + "="*80)
        print("📋 تقرير أداء النموذج الشامل")
        print("="*80)
        print(f"\n{report['summary']}")
        
        if save_to_file:
            filename = f"model_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n💾 تم حفظ التقرير في: {filename}")
        
        return report
    
    def _generate_performance_summary(self):
        """توليد ملخص الأداء"""
        metrics = self.performance_metrics
        
        summary = f"""
🎯 ملخص أداء نموذج التنبؤ:

• الدقة الشاملة: {metrics['overall_accuracy']:.1%}
• إجمالي التنبؤات المُقيمة: {metrics['total_predictions']}

📊 تفصيل الدقة:
  • توقع النتيجة: {metrics['result_accuracy']:.1%}
  • توقع الفوز/التعادل/الخسارة: {metrics['outcome_accuracy']:.1%}
  • توقع النتيجة الدقيقة: {metrics['score_accuracy']:.1%}
  • توقع فارق الأهداف: {metrics['goal_difference_accuracy']:.1%}
  • توقع أكثر/أقل من 2.5 هدف: {metrics['over_under_accuracy']:.1%}
  • توقع تسجيل الفريقين: {metrics['both_teams_score_accuracy']:.1%}

📈 التقييم: {'ممتاز' if metrics['overall_accuracy'] > 0.6 else 'جيد' if metrics['overall_accuracy'] > 0.5 else 'بحاجة للتحسين'}
"""
        return summary
    
    def compare_with_baseline(self, baseline_method='random'):
        """مقارنة أداء النموذج مع طرق baseline"""
        print(f"\n🔍 مقارنة مع {baseline_method} baseline...")
        
        if baseline_method == 'random':
            # دقة عشوائية نظرية
            baseline_accuracy = 0.33  # 33% لثلاث نتائج محتملة
        elif baseline_method == 'home_win_bias':
            # افتراض فوز الفريق المنزل دائماً
            home_win_rate = len(self.data[self.data['FTR'] == 'H']) / len(self.data)
            baseline_accuracy = home_win_rate
        else:
            baseline_accuracy = 0.33
        
        model_accuracy = self.performance_metrics.get('result_accuracy', 0)
        improvement = ((model_accuracy - baseline_accuracy) / baseline_accuracy) * 100
        
        print(f"• دقة النموذج: {model_accuracy:.1%}")
        print(f"• دقة {baseline_method} baseline: {baseline_accuracy:.1%}")
        print(f"• التحسن: {improvement:+.1f}%")
        
        if improvement > 0:
            print("✅ النموذج يتفوق على baseline")
        else:
            print("❌ النموذج يحتاج تحسين لتجاوز baseline")

# مثال على الاستخدام
if __name__ == "__main__":
    # اختبار الكلاس المحسن
    validator = DataValidator("data/football-data/combined_seasons_data.csv")
    
    if validator.validate_and_clean_data():
        print("\n✅ اكتمل التحقق من البيانات بنجاح!")
        
        # محاكاة تسجيل بعض التنبؤات
        sample_predictions = [
            {
                'id': 'pred_001',
                'prediction': {
                    'multiple_predictions': [
                        {'home_goals': 2, 'away_goals': 1, 'probability': 0.15, 'type': 'الأكثر ترجيحاً'}
                    ],
                    'probabilities': {
                        'home_win': 0.55, 'draw': 0.25, 'away_win': 0.20,
                        'over_2_5': 0.65, 'both_teams_score': 0.70
                    }
                },
                'match_info': {
                    'home_team': 'Man City',
                    'away_team': 'Liverpool', 
                    'date': '2023-05-15'
                }
            }
        ]
        
        for pred in sample_predictions:
            validator.record_prediction(pred['id'], pred['prediction'], pred['match_info'])
        
        # التحقق من دقة التنبؤات
        accuracy_results = validator.validate_predictions_accuracy()
        
        if accuracy_results:
            # تقييم أداء النموذج
            validator.evaluate_model_performance()
            
            # إنشاء تقرير شامل
            validator.generate_performance_report(save_to_file=True)
            
            # مقارنة مع baseline
            validator.compare_with_baseline('random')