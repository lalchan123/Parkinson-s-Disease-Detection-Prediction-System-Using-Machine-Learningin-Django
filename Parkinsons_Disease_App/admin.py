from django.contrib import admin
from .models import ParkinsonPredict
# Register your models here.
class ParkinsonPredictAdmin(admin.ModelAdmin):
    list_display = ('MDVP_Fo','MDVP_Fhi','MDVP_Flo','MDVP_Jitter','MDVP_Jitter_Abs','MDVP_RAP','status')
    list_filter = ('status',)

admin.site.register(ParkinsonPredict,ParkinsonPredictAdmin)