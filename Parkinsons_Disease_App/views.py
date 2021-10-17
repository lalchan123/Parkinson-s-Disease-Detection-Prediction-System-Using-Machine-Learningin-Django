from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from .models import ParkinsonPredict
from django.core.paginator import Paginator

# Create your views here.
def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")    


def result(request): 
    parkinsons_data = pd.read_csv(r'F:\Project and Thesis\Project\Django Project\ML Django Project\Parkinson_Disease_Detection Prediction\Parkinsons_Disease_Prediction\parkinsons.data')
    X = parkinsons_data.drop(columns=['name','status'], axis=1)
    Y = parkinsons_data['status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    model = svm.SVC(kernel='linear')
    model.fit(X_test, Y_test)
    if request.method == "POST":
        n1 = float(request.POST['n1'])
        n2 = float(request.POST['n2'])
        n3 = float(request.POST['n3'])
        n4 = float(request.POST['n4'])
        n5 = float(request.POST['n5'])
        n6 = float(request.POST['n6'])
        n7 = float(request.POST['n7'])
        n8 = float(request.POST['n8'])
        n9 = float(request.POST['n9'])
        n10 = float(request.POST['n10'])
        n11 = float(request.POST['n11'])
        n12 = float(request.POST['n12'])
        n13 = float(request.POST['n13'])
        n14 = float(request.POST['n14'])
        n15 = float(request.POST['n15'])
        n16 = float(request.POST['n16'])
        n17 = float(request.POST['n17'])
        n18 = float(request.POST['n18'])
        n19 = float(request.POST['n19'])
        n20 = float(request.POST['n20'])
        n21 = float(request.POST['n21'])
        n22 = float(request.POST['n22'])
        pred = model.predict(np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22]).reshape(1,-1))
        result = ""
        if pred == [1]:
            result = "The Person has Parkinsons"
        else:
            result = "The Person has not Parkinsons"
        
        data = ParkinsonPredict(MDVP_Fo=n1,MDVP_Fhi=n2,MDVP_Flo=n3,MDVP_Jitter=n4,MDVP_Jitter_Abs=n5,MDVP_RAP=n6,MDVP_PPQ=n7,Jitter_DDP=n8,MDVP_Shimmer=n9,MDVP_Shimmer_dB=n10,Shimmer_APQ3=n11,Shimmer_APQ5=n12,MDVP_APQ=n13,Shimmer_DDA=n14,NHR=n15,HNR=n16,RPDE=n17,DFA=n18,spread1=n19,spread2=n20,D2=n21,PPE=n22,status=result)
        data.save()
    
    return render(request, "predict.html", {"result":result,})

def recordData(request):
    data = ParkinsonPredict.objects.all().order_by('-id')
    paginator = Paginator(data, 10) 
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, "recordData.html",{"page_obj":page_obj, })  
