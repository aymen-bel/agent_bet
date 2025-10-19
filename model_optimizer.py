# model_optimizer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    def __init__(self, historical_data_file, predictions_data=None):
        self.historical_data = pd.read_csv(historical_data_file)
        self.predictions_data = predictions_data
        self.optimized_weights = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def preprocess_historical_data(self):
        """معالجة البيانات التاريخية للتدريب"""
        print("🔧 جاري معالجة البيانات التاريخية...")
        
        # تنظيف البيانات الأساسية
        self.historical_data['HomeTeam'] = self.historical_data['HomeTeam'].astype(str).str.strip()
        self.historical_data['AwayTeam'] = self.historical_data['AwayTeam'].astype(str).str.strip()
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'], dayfirst=True, errors='coerce')
        
        # إزالة المباريات غير المكتملة
        required_columns = ['FTHG', 'FTAG', 'FTR', 'HomeTeam', 'AwayTeam']
        self.historical_data = self.historical_data.dropna(subset=required_columns)
        
        # إنشاء متغير الهدف
        self.historical_data['Result'] = self.historical_data['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        self.historical_data['Total_Goals'] = self.historical_data['FTHG'] + self.historical_data['FTAG']
        self.historical_data['Goal_Difference'] = self.historical_data['FTHG'] - self.historical_data['FTAG']
        
        print(f"✅ تمت معالجة {len(self.historical_data)} مباراة تاريخية")
        return self.historical_data
    
    def calculate_match_features(self, home_team, away_team, match_date):
        """حساب ميزات المباراة بناءً على البيانات التاريخية حتى تاريخ المباراة"""
        # بيانات الفريق المنزل حتى تاريخ المباراة
        home_data = self.historical_data[
            ((self.historical_data['HomeTeam'] == home_team) | 
             (self.historical_data['AwayTeam'] == home_team)) &
            (self.historical_data['Date'] < match_date)
        ]
        
        # بيانات الفريق الضيف حتى تاريخ المباراة
        away_data = self.historical_data[
            ((self.historical_data['HomeTeam'] == away_team) | 
             (self.historical_data['AwayTeam'] == away_team)) &
            (self.historical_data['Date'] < match_date)
        ]
        
        if len(home_data) < 5 or len(away_data) < 5:
            return None
        
        # حساب ميزات الفريق المنزل
        home_features = self._calculate_team_features(home_data, home_team, 'home')
        
        # حساب ميزات الفريق الضيف
        away_features = self._calculate_team_features(away_data, away_team, 'away')
        
        # ميزات المواجهات المباشرة
        head_to_head = self._calculate_head_to_head_features(home_team, away_team, match_date)
        
        # الجمع بين جميع الميزات
        match_features = {**home_features, **away_features, **head_to_head}
        
        return match_features
    
    def _calculate_team_features(self, team_data, team_name, team_type):
        """حساب ميزات الفريق"""
        home_matches = team_data[team_data['HomeTeam'] == team_name]
        away_matches = team_data[team_data['AwayTeam'] == team_name]
        
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches == 0:
            return self._get_default_features(team_type)
        
        # الإحصائيات الأساسية
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        draws = len(home_matches[home_matches['FTR'] == 'D']) + len(away_matches[away_matches['FTR'] == 'D'])
        
        total_points = (home_wins + away_wins) * 3 + draws
        goals_scored = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        goals_conceded = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        
        # الميزات المتقدمة
        features = {
            f'{team_type}_matches': total_matches,
            f'{team_type}_points': total_points,
            f'{team_type}_goals_scored': goals_scored,
            f'{team_type}_goals_conceded': goals_conceded,
            f'{team_type}_goal_difference': goals_scored - goals_conceded,
            f'{team_type}_win_rate': (home_wins + away_wins) / total_matches,
            f'{team_type}_points_per_match': total_points / total_matches,
            f'{team_type}_goals_per_match': goals_scored / total_matches,
            f'{team_type}_conceded_per_match': goals_conceded / total_matches,
        }
        
        # إضافة إحصائيات التصويب إذا كانت متاحة
        if 'HS' in home_matches.columns and 'AS' in away_matches.columns:
            home_shots = home_matches['HS'].sum()
            away_shots = away_matches['AS'].sum()
            total_shots = home_shots + away_shots
            
            home_shots_target = home_matches['HST'].sum() if 'HST' in home_matches.columns else home_shots * 0.35
            away_shots_target = away_matches['AST'].sum() if 'AST' in away_matches.columns else away_shots * 0.35
            total_shots_target = home_shots_target + away_shots_target
            
            features.update({
                f'{team_type}_shot_efficiency': goals_scored / total_shots if total_shots > 0 else 0,
                f'{team_type}_shot_accuracy': total_shots_target / total_shots if total_shots > 0 else 0,
                f'{team_type}_shots_per_match': total_shots / total_matches,
            })
        
        return features
    
    def _calculate_head_to_head_features(self, home_team, away_team, match_date):
        """حساب ميزات المواجهات المباشرة"""
        h2h_data = self.historical_data[
            ((self.historical_data['HomeTeam'] == home_team) & (self.historical_data['AwayTeam'] == away_team)) |
            ((self.historical_data['HomeTeam'] == away_team) & (self.historical_data['AwayTeam'] == home_team))
        ]
        h2h_data = h2h_data[h2h_data['Date'] < match_date]
        
        if len(h2h_data) == 0:
            return {
                'h2h_matches': 0,
                'h2h_home_wins': 0,
                'h2h_away_wins': 0,
                'h2h_draws': 0,
                'h2h_home_advantage': 0.5
            }
        
        home_wins = len(h2h_data[
            ((h2h_data['HomeTeam'] == home_team) & (h2h_data['FTR'] == 'H')) |
            ((h2h_data['HomeTeam'] == away_team) & (h2h_data['FTR'] == 'A'))
        ])
        
        away_wins = len(h2h_data[
            ((h2h_data['HomeTeam'] == home_team) & (h2h_data['FTR'] == 'A')) |
            ((h2h_data['HomeTeam'] == away_team) & (h2h_data['FTR'] == 'H'))
        ])
        
        draws = len(h2h_data[h2h_data['FTR'] == 'D'])
        
        return {
            'h2h_matches': len(h2h_data),
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_draws': draws,
            'h2h_home_advantage': home_wins / len(h2h_data) if len(h2h_data) > 0 else 0.5
        }
    
    def _get_default_features(self, team_type):
        """قيم افتراضية عندما لا توجد بيانات كافية"""
        return {
            f'{team_type}_matches': 1,
            f'{team_type}_points': 1.5,
            f'{team_type}_goals_scored': 1.5,
            f'{team_type}_goals_conceded': 1.5,
            f'{team_type}_goal_difference': 0,
            f'{team_type}_win_rate': 0.33,
            f'{team_type}_points_per_match': 1.5,
            f'{team_type}_goals_per_match': 1.5,
            f'{team_type}_conceded_per_match': 1.5,
            f'{team_type}_shot_efficiency': 0.1,
            f'{team_type}_shot_accuracy': 0.35,
            f'{team_type}_shots_per_match': 12,
        }
    
    def prepare_training_data(self, min_matches=100):
        """تحضير بيانات التدريب من المباريات التاريخية"""
        print("📊 جاري تحضير بيانات التدريب...")
        
        features_list = []
        results_list = []
        goals_list = []
        
        # ترتيب المباريات حسب التاريخ
        historical_sorted = self.historical_data.sort_values('Date')
        
        for idx, match in historical_sorted.iterrows():
            if idx < min_matches:  # تخطي المباريات الأولى لضمان وجود بيانات كافية
                continue
                
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            match_date = match['Date']
            
            # حساب الميزات
            match_features = self.calculate_match_features(home_team, away_team, match_date)
            
            if match_features:
                features_list.append(match_features)
                results_list.append(match['Result'])
                goals_list.append({
                    'home_goals': match['FTHG'],
                    'away_goals': match['FTAG'],
                    'total_goals': match['FTHG'] + match['FTAG']
                })
        
        # تحويل إلى DataFrame
        features_df = pd.DataFrame(features_list)
        results_df = pd.Series(results_list)
        goals_df = pd.DataFrame(goals_list)
        
        print(f"✅ تم تحضير {len(features_df)} عينة للتدريب")
        return features_df, results_df, goals_df
    
    def train_result_predictor(self, features_df, results_df):
        """تدريب نموذج للتنبؤ بنتيجة المباراة (فوز/تعادل/خسارة)"""
        print("🤖 جاري تدريب نموذج تنبؤ النتائج...")
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, results_df, test_size=0.2, random_state=42, stratify=results_df
        )
        
        # تدريب عدة نماذج
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            # تدريب النموذج
            model.fit(X_train, y_train)
            
            # التقيم
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # التقيم المتقاطع
            cv_scores = cross_val_score(model, features_df, results_df, cv=5)
            cv_mean = cv_scores.mean()
            
            print(f"📊 {name}:")
            print(f"   • دقة الاختبار: {accuracy:.3f}")
            print(f"   • متوسط التقيم المتقاطع: {cv_mean:.3f}")
            
            if cv_mean > best_score:
                best_score = cv_mean
                best_model = model
                best_model_name = name
        
        # حفظ أفضل نموذج
        self.result_predictor = best_model
        self.model_performance['result_predictor'] = {
            'model': best_model_name,
            'accuracy': best_score,
            'feature_importance': self._get_feature_importance(best_model, features_df.columns)
        }
        
        print(f"✅ أفضل نموذج للنتائج: {best_model_name} (دقة: {best_score:.3f})")
        return best_model
    
    def train_goals_predictor(self, features_df, goals_df):
        """تدريب نموذج للتنبؤ بعدد الأهداف"""
        print("🎯 جاري تدريب نموذج تنبؤ الأهداف...")
        
        # نماذج التنبؤ بالأهداف
        models = {
            'Home Goals': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Away Goals': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Total Goals': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        goals_predictors = {}
        
        for target, model in models.items():
            if target == 'Home Goals':
                y = goals_df['home_goals']
            elif target == 'Away Goals':
                y = goals_df['away_goals']
            else:
                y = goals_df['total_goals']
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, y, test_size=0.2, random_state=42
            )
            
            # تدريب النموذج
            model.fit(X_train, y_train)
            
            # التقيم
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            # التقيم المتقاطع
            cv_scores = -cross_val_score(model, features_df, y, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = cv_scores.mean()
            
            print(f"📊 {target}:")
            print(f"   • متوسط الخطأ المطلق: {mae:.3f}")
            print(f"   • متوسط التقيم المتقاطع: {cv_mae:.3f}")
            
            goals_predictors[target] = model
            
            self.model_performance[f'{target.lower().replace(" ", "_")}_predictor'] = {
                'model': 'GradientBoostingRegressor',
                'mae': cv_mae,
                'feature_importance': self._get_feature_importance(model, features_df.columns)
            }
        
        self.goals_predictor = goals_predictors
        return goals_predictors
    
    def _get_feature_importance(self, model, feature_names):
        """استخراج أهمية الميزات من النموذج"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                return {}
            
            # تطبيع الأهمية
            importance = importance / importance.sum()
            feature_importance = dict(zip(feature_names, importance))
            
            # ترتيب حسب الأهمية
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
        except:
            return {}
    
    def analyze_prediction_errors(self, predictions_file, actual_results_file):
        """تحليل الفروق بين التوقعات والنتائج الفعلية"""
        print("🔍 جاري تحليل أخطاء التوقع...")
        
        # تحميل التوقعات والنتائج
        predictions = pd.read_csv(predictions_file)
        actual_results = pd.read_csv(actual_results_file)
        
        # دمج البيانات
        merged_data = pd.merge(predictions, actual_results, 
                              on=['HomeTeam', 'AwayTeam', 'Date'], 
                              suffixes=('_pred', '_actual'))
        
        # حساب الأخطاء
        errors = []
        
        for _, match in merged_data.iterrows():
            # خطأ نتيجة المباراة
            result_pred = match.get('Predicted_Result', 'H')
            result_actual = match.get('FTR', 'H')
            result_error = 1 if result_pred != result_actual else 0
            
            # خطأ عدد الأهداف
            home_goals_pred = match.get('Predicted_Home_Goals', 0)
            away_goals_pred = match.get('Predicted_Away_Goals', 0)
            home_goals_actual = match['FTHG']
            away_goals_actual = match['FTAG']
            
            goals_error = abs(home_goals_pred - home_goals_actual) + abs(away_goals_pred - away_goals_actual)
            
            errors.append({
                'home_team': match['HomeTeam'],
                'away_team': match['AwayTeam'],
                'date': match['Date'],
                'result_error': result_error,
                'goals_error': goals_error,
                'predicted_result': result_pred,
                'actual_result': result_actual,
                'predicted_home_goals': home_goals_pred,
                'actual_home_goals': home_goals_actual,
                'predicted_away_goals': away_goals_pred,
                'actual_away_goals': away_goals_actual
            })
        
        errors_df = pd.DataFrame(errors)
        
        # إحصائيات الأخطاء
        total_matches = len(errors_df)
        result_accuracy = 1 - errors_df['result_error'].mean()
        avg_goals_error = errors_df['goals_error'].mean()
        
        print(f"📈 إحصائيات دقة التوقعات:")
        print(f"   • دقة توقع النتيجة: {result_accuracy:.1%}")
        print(f"   • متوسط خطأ الأهداف: {avg_goals_error:.2f}")
        print(f"   • إجمالي المباريات: {total_matches}")
        
        # تحليل أنماط الأخطاء
        self._analyze_error_patterns(errors_df)
        
        return errors_df
    
    def _analyze_error_patterns(self, errors_df):
        """تحليل أنماط الأخطاء الشائعة"""
        print("\n🔎 تحليل أنماط الأخطاء:")
        
        # الأخطاء حسب نوع النتيجة
        result_errors = errors_df[errors_df['result_error'] == 1]
        
        if len(result_errors) > 0:
            print("📊 توزيع أخطاء النتائج:")
            error_patterns = result_errors.groupby(['predicted_result', 'actual_result']).size()
            for (pred, actual), count in error_patterns.items():
                print(f"   • {pred} → {actual}: {count} مباراة")
        
        # المباريات ذات الخطأ الكبير في الأهداف
        high_goal_errors = errors_df[errors_df['goals_error'] > 2]
        if len(high_goal_errors) > 0:
            print(f"\n⚠️  المباريات ذات الخطأ الكبير في الأهداف (>2): {len(high_goal_errors)}")
            for _, match in high_goal_errors.head().iterrows():
                print(f"   • {match['home_team']} {match['predicted_home_goals']}-{match['predicted_away_goals']} {match['away_team']} "
                      f"(الفعلي: {match['actual_home_goals']}-{match['actual_away_goals']})")
    
    def optimize_weights_based_on_errors(self, errors_df, team_assessment_data):
        """تحسين الأوزان بناءً على تحليل الأخطاء"""
        print("\n🔄 جاري تحسين الأوزان بناءً على تحليل الأخطاء...")
        
        # تحليل المباريات التي حدثت فيها أخطاء كبيرة
        high_error_matches = errors_df[errors_df['goals_error'] > 2]
        
        if len(high_error_matches) == 0:
            print("✅ لا توجد أخطاء كبيرة لتحسين الأوزان")
            return self.optimized_weights
        
        weight_adjustments = {}
        
        for _, match in high_error_matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # تحليل سبب الخطأ
            error_analysis = self._analyze_single_error(match, team_assessment_data)
            
            for feature, adjustment in error_analysis.items():
                if feature not in weight_adjustments:
                    weight_adjustments[feature] = []
                weight_adjustments[feature].append(adjustment)
        
        # حساب متوسط التعديلات
        for feature, adjustments in weight_adjustments.items():
            avg_adjustment = np.mean(adjustments)
            weight_adjustments[feature] = max(0.1, min(2.0, 1.0 + avg_adjustment))
        
        self.optimized_weights = weight_adjustments
        
        print("✅ الأوزان المحسنة:")
        for feature, weight in sorted(weight_adjustments.items(), key=lambda x: abs(x[1]-1), reverse=True)[:10]:
            change = "↑" if weight > 1 else "↓"
            print(f"   • {feature}: {weight:.2f} ({change})")
        
        return weight_adjustments
    
    def _analyze_single_error(self, match_error, team_assessment_data):
        """تحليل خطأ في مباراة فردية"""
        adjustments = {}
        
        home_team = match_error['home_team']
        away_team = match_error['away_team']
        
        # الحصول على بيانات الفريقين
        home_data = team_assessment_data.get(home_team, {})
        away_data = team_assessment_data.get(away_team, {})
        
        # تحليل الفروق بين التوقع والواقع
        predicted_home_goals = match_error['predicted_home_goals']
        actual_home_goals = match_error['actual_home_goals']
        predicted_away_goals = match_error['predicted_away_goals']
        actual_away_goals = match_error['actual_away_goals']
        
        home_goal_error = actual_home_goals - predicted_home_goals
        away_goal_error = actual_away_goals - predicted_away_goals
        
        # تحديد المجالات التي تحتاج لتعديل
        if home_goal_error > 0:  # تم التقليل من هجوم المنزل
            if home_data.get('shot_efficiency', 0) > 0.3:
                adjustments['shot_efficiency'] = 0.1
            if home_data.get('conversion_rate', 0) > 0.3:
                adjustments['conversion_rate'] = 0.1
        
        if away_goal_error > 0:  # تم التقليل من هجوم الضيف
            if away_data.get('shot_efficiency', 0) > 0.3:
                adjustments['shot_efficiency'] = 0.1
            if away_data.get('conversion_rate', 0) > 0.3:
                adjustments['conversion_rate'] = 0.1
        
        if home_goal_error < 0:  # تم المبالغة في هجوم المنزل
            if home_data.get('defensive_efficiency', 0) < 0.6:
                adjustments['defensive_efficiency'] = -0.1
        
        if away_goal_error < 0:  # تم المبالغة في هجوم الضيف
            if away_data.get('defensive_efficiency', 0) < 0.6:
                adjustments['defensive_efficiency'] = -0.1
        
        return adjustments
    
    def generate_optimization_report(self):
        """توليد تقرير تحسين شامل"""
        print("\n" + "="*80)
        print("📊 تقرير تحسين النموذج الشامل")
        print("="*80)
        
        print(f"\n🎯 أداء النماذج:")
        for model_name, performance in self.model_performance.items():
            if 'accuracy' in performance:
                print(f"• {model_name}: {performance['accuracy']:.3f} دقة")
            elif 'mae' in performance:
                print(f"• {model_name}: {performance['mae']:.3f} متوسط خطأ")
        
        print(f"\n🔍 أهم المؤشرات المؤثرة:")
        for model_name, performance in self.model_performance.items():
            if 'feature_importance' in performance and performance['feature_importance']:
                top_features = list(performance['feature_importance'].items())[:5]
                print(f"\n• {model_name}:")
                for feature, importance in top_features:
                    print(f"  - {feature}: {importance:.3f}")
        
        if self.optimized_weights:
            print(f"\n🔄 تعديلات الأوزان الموصى بها:")
            for feature, weight in sorted(self.optimized_weights.items(), 
                                        key=lambda x: abs(x[1]-1), reverse=True)[:10]:
                change_percent = (weight - 1) * 100
                change_dir = "زيادة" if change_percent > 0 else "تقليل"
                print(f"• {feature}: {change_dir} بنسبة {abs(change_percent):.1f}%")
        
        print(f"\n💡 توصيات التحسين:")
        recommendations = [
            "التركيز على مؤشرات الهجوم للفرق ذات الكفاءة العالية في التسجيل",
            "مراجعة تقدير القوة الدفاعية للفرق في المباريات الصعبة",
            "إضافة مؤشرات الشكل الحالي (آخر 5 مباريات) لتحسين الدقة",
            "مراعاة العوامل الخارجية مثل الإصابات والإيقافات"
        ]
        
        for i, recommendation in enumerate(recommendations, 1):
            print(f"  {i}. {recommendation}")
    
    def save_optimized_model(self, filename="optimized_match_predictor.pkl"):
        """حفظ النموذج المحسن"""
        model_data = {
            'result_predictor': self.result_predictor,
            'goals_predictor': self.goals_predictor,
            'optimized_weights': self.optimized_weights,
            'model_performance': self.model_performance,
            'timestamp': datetime.now()
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ تم حفظ النموذج المحسن في {filename}")
    
    def load_optimized_model(self, filename="optimized_match_predictor.pkl"):
        """تحميل النموذج المحسن"""
        try:
            model_data = joblib.load(filename)
            self.result_predictor = model_data['result_predictor']
            self.goals_predictor = model_data['goals_predictor']
            self.optimized_weights = model_data['optimized_weights']
            self.model_performance = model_data['model_performance']
            print(f"✅ تم تحميل النموذج المحسن من {filename}")
            return True
        except:
            print(f"❌ تعذر تحميل النموذج من {filename}")
            return False

# نموذج Gradient Boosting للتصنيف
from sklearn.ensemble import GradientBoostingClassifier

# التشغيل الرئيسي
if __name__ == "__main__":
    print("🚀 بدء نظام تحسين النموذج...")
    
    try:
        # تحميل البيانات التاريخية
        optimizer = ModelOptimizer("data/football-data/combined_seasons_data.csv")
        
        # معالجة البيانات
        optimizer.preprocess_historical_data()
        
        # تحضير بيانات التدريب
        features_df, results_df, goals_df = optimizer.prepare_training_data()
        
        if len(features_df) > 0:
            # تدريب نماذج التنبؤ
            result_model = optimizer.train_result_predictor(features_df, results_df)
            goals_models = optimizer.train_goals_predictor(features_df, goals_df)
            
            # حفظ النماذج
            optimizer.save_optimized_model()
            
            # توليد التقرير
            optimizer.generate_optimization_report()
            
            print(f"\n✅ تم الانتهاء من تحسين النموذج بنجاح!")
            print(f"📊 تم تدريب النماذج على {len(features_df)} مباراة تاريخية")
            
        else:
            print("❌ لا توجد بيانات كافية لتدريب النماذج")
            
    except Exception as e:
        print(f"❌ حدث خطأ: {e}")
        import traceback
        traceback.print_exc()