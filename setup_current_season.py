# setup_current_season.py
import pandas as pd
import os
from data.download_seasons import SeasonDataDownloader

def setup_complete_system():
    """إعداد النظام الكامل مع البيانات الحقيقية"""
    print("🏗️ بدء إعداد النظام ببيانات الموسم الحالي 2025/2026...")
    
    # 1. تحميل بيانات جميع المواسم
    downloader = SeasonDataDownloader()
    downloader.download_all_seasons()
    
    # 2. التحقق من جودة البيانات
    validate_data_quality()
    
    # 3. إنشاء ملف التكامل الزمني
    create_temporal_integration_file()
    
    print("✅ اكتمل إعداد النظام ببيانات حقيقية!")
    print("📊 إحصائيات البيانات:")
    print_stats()

def validate_data_quality():
    """التحقق من جودة البيانات المحملة"""
    seasons = ['2020', '2021', '2022', '2023', '2024', '2025']
    
    for year in seasons:
        filepath = f"data/seasons/england_E0_{year}.csv"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"✅ موسم {year}: {len(df)} مباراة")
        else:
            print(f"❌ موسم {year}: ملف غير موجود")

def create_temporal_integration_file():
    """إنشاء ملف تكامل زمني لجميع المواسم"""
    all_seasons = []
    
    for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
        filepath = f"data/seasons/england_E0_{year}.csv"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Season'] = year
            all_seasons.append(df)
    
    if all_seasons:
        combined_df = pd.concat(all_seasons, ignore_index=True)
        combined_df.to_csv("data/combined_seasons_2020_2025.csv", index=False)
        print(f"✅ تم دمج {len(combined_df)} مباراة من 6 مواسم")

def print_stats():
    """طباعة إحصائيات البيانات"""
    try:
        df_2025 = pd.read_csv("data/seasons/england_E0_2025.csv")
        played_matches = df_2025[df_2025['FTHG'].notna()]
        upcoming_matches = df_2025[df_2025['FTHG'].isna()]
        
        print(f"🎯 الموسم الحالي 2025/2026:")
        print(f"   • المباريات المنتهية: {len(played_matches)}")
        print(f"   • المباريات القادمة: {len(upcoming_matches)}")
        print(f"   • إجمالي المباريات: {len(df_2025)}")
        
    except Exception as e:
        print(f"❌ خطأ في قراءة إحصائيات 2025: {e}")

if __name__ == "__main__":
    setup_complete_system()