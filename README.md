# stash
# K-water Project

# 수상 불가 이유
 - 본 프로젝트에서 사용한 방법은 XGBoost
 - 대회에서 요구한 조건은 t-n ~ t-1의 데이터를 통해 t ~ t+335의 336개 시점 데이터를 모두 예측하는 것
 - 하지만, xgboost의 X_test를 사용할 때 336개 시점의 데이터를 한번에 뽑지않고 예측 시점의 값을 예측하기 위해 해당 시점 - 1 의 값을 사용했던 것이 문제
   정확하게 하기 위해서는 t-n ~ t-1 의 데이터를 통해 t를 예측하고 예측한 t를 포함한 것으로 t+1을 예측, 예측한 t+1을 포함하고 t+2를 예측하는 이러한 방식으로 문제를 해결해야 헀음
   이런 방식을 사용하면 336개의 시점을 예측하고자 할 때, 미래의 데이터를 사용하지 않고 예측할 수 있게 되는 것
