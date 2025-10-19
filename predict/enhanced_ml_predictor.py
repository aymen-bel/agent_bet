import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

class EnhancedMLPredictor:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
    def prepare_advanced_features(self, historical_data):
        """تحضير ميزات متقدمة للتدريب"""
        print("🔄 تحضير الميزات المتقدمة...")
        
        features_list = []
        targets_result = []
        targets_home_goals = []
        targets_away_goals = []
        
        for idx, match in historical_data.iterrows():
            # استخدام البيانات حتى تاريخ المباراة فقط
            match_date = match['Date']
            past_data = historical_data[historical_data['Date'] < match_date]
            
            home_features = self._get_team_features(past_data, match['HomeTeam'], match_date, 'home')
            away_features = self._get_team_features(past_data, match['AwayTeam'], match_date, 'away')
            
            if home_features and away_features:
                match_features = {**home_features, **away_features}
                features_list.append(match_features)
                
                # الأهداف
                targets_result.append(match['FTR'])
                targets_home_goals.append(match['FTHG'])
                targets_away_goals.append(match['FTAG'])
        
        features_df = pd.DataFrame(features_list)
        
        print(f"✅ تم تحضير {len(features_df)} عينة بميزات متقدمة")
        return features_df, targets_result, targets_home_goals, targets_away_goals
    
    def _get_team_features(self, data, team_name, date, venue):
        """الحصول على ميزات الفريق حتى تاريخ محدد"""
        team_data = data[(data['HomeTeam'] == team_name) | (data['AwayTeam'] == team_name)]
        team_data = team_data[team_data['Date'] < date]
        
        if len(team_data) < 5:
            return None
        
        home_matches = team_data[team_data['HomeTeam'] == team_name]
        away_matches = team_data[team_data['AwayTeam'] == team_name]
        
        # الميزات الأساسية
        features = {
            f'{venue}_matches': len(team_data),
            f'{venue}_win_rate': (len(home_matches[home_matches['FTR'] == 'H']) + 
                                len(away_matches[away_matches['FTR'] == 'A'])) / len(team_data),
            f'{venue}_goals_for': home_matches['FTHG'].sum() + away_matches['FTAG'].sum(),
            f'{venue}_goals_against': home_matches['FTAG'].sum() + away_matches['FTHG'].sum(),
            f'{venue}_form': self._calculate_recent_form(team_data.tail(5), team_name)
        }
        
        # إضافة ميزات التصويب إذا كانت متاحة
        if all(col in team_data.columns for col in ['HS', 'HST']):
            total_shots = home_matches['HS'].sum() + away_matches['AS'].sum()
            if total_shots > 0:
                features[f'{venue}_shot_efficiency'] = (home_matches['FTHG'].sum() + away_matches['FTAG'].sum()) / total_shots
            else:
                features[f'{venue}_shot_efficiency'] = 0.1
        
        return features

    def _calculate_recent_form(self, recent_matches, team_name):
        """حساب النقاط في آخر 5 مباريات"""
        points = 0
        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team_name:
                if match['FTR'] == 'H':
                    points += 3
                elif match['FTR'] == 'D':
                    points += 1
            else:  # Away
                if match['FTR'] == 'A':
                    points += 3
                elif match['FTR'] == 'D':
                    points += 1
        return points / (len(recent_matches) * 3)  # نسبة النقاط من إجمالي النقاط الممكنة
    
    def train_advanced_models(self, features_df, targets_result, targets_home_goals, targets_away_goals):
        """تدريب نماذج متقدمة"""
        print("🤖 تدريب النماذج المتقدمة...")
        
        # تحويل النتائج إلى أرقام
        result_mapping = {'H': 0, 'D': 1, 'A': 2}
        targets_result_numeric = [result_mapping[r] for r in targets_result]
        
        # تقسيم البيانات
        X_train, X_test, y_result_train, y_result_test = train_test_split(
            features_df, targets_result_numeric, test_size=0.2, random_state=42, stratify=targets_result_numeric
        )
        
        # نموذج النتائج
        result_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        result_model.fit(X_train, y_result_train)
        y_pred = result_model.predict(X_test)
        accuracy = accuracy_score(y_result_test, y_pred)
        
        # التقيم المتقاطع
        cv_scores = cross_val_score(result_model, features_df, targets_result_numeric, cv=5)
        
        print(f"🎯 نموذج النتائج - الدقة: {accuracy:.3f}, التقيم المتقاطع: {cv_scores.mean():.3f}")
        
        # نماذج الأهداف
        home_goals_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        away_goals_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        X_train_goals, X_test_goals, y_home_train, y_home_test = train_test_split(
            features_df, targets_home_goals, test_size=0.2, random_state=42
        )
        _, _, y_away_train, y_away_test = train_test_split(
            features_df, targets_away_goals, test_size=0.2, random_state=42
        )
        
        home_goals_model.fit(X_train_goals, y_home_train)
        away_goals_model.fit(X_train_goals, y_away_train)
        
        home_mae = mean_absolute_error(y_home_test, home_goals_model.predict(X_test_goals))
        away_mae = mean_absolute_error(y_away_test, away_goals_model.predict(X_test_goals))
        
        print(f"⚽ نموذج الأهداف - خطأ المنزل: {home_mae:.3f}, خطأ الضيف: {away_mae:.3f}")
        
        self.models = {
            'result': result_model,
            'home_goals': home_goals_model,
            'away_goals': away_goals_model
        }
        
        self.feature_importance = {
            'result': dict(zip(features_df.columns, result_model.feature_importances_)),
            'home_goals': dict(zip(features_df.columns, home_goals_model.feature_importances_)),
            'away_goals': dict(zip(features_df.columns, away_goals_model.feature_importances_))
        }
        
        return self.models