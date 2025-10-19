# examples/betting_demo.py

from advanced_system import AdvancedBettingSystem
import json
from datetime import datetime

def demo_comprehensive_betting():
    """عرض توضيحي لنظام الرهانات الشامل"""
    print("🎰 العرض التوضيحي لنظام الرهانات الشامل")
    print("=" * 50)
    
    # تحميل النظام المدرب
    system = AdvancedBettingSystem("data/football-data")
    
    # محاولة تحميل نموذج مدرب إذا موجود
    try:
        # هنا يمكنك تحميل نموذج مدرب مسبقاً
        print("📂 جاري تحميل النظام المدرب...")
    except:
        print("⚠️  النظام غير مدرب، سيتم استخدام الوضع الافتراضي")
    
    # تحليل شامل للرهانات
    print("\n🔍 جاري تحليل أداء الرهانات...")
    betting_analysis = system.comprehensive_betting_analysis(stake=50.0)
    
    # عرض التقرير
    if betting_analysis:
        report = system.get_detailed_betting_report(betting_analysis)
        print("\n" + report)
        
        # حفظ النتائج
        with open(f"betting_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
            json.dump(betting_analysis, f, ensure_ascii=False, indent=2)
        print(f"\n💾 تم حفظ التحليل الكامل في ملف JSON")
    else:
        print("❌ فشل في تحليل الرهانات")

if __name__ == "__main__":
    demo_comprehensive_betting()