install.packages("package name") # package install
library(library name) # include library
head(kyphosis) # data frame의 일부 정보를 확인할 때 사용

df = cbind(df, cn) # df에 column cn을 붙일 때 사용한다.
df = rbind(df, cn) # df에 row cn을 붙일 때 사용한다.

names(df)[1] <- c("new column name") # df의 1번 컬럼의 이름을 바꾼다. R은 index가 1부터 시작한다.
names(df) # df의 column names 출력

library(dplyr) # 다양한 함수들을 제공하는 라이브러리
df = select(kyphosis, Age, Number) # df kyphosis에서 Age, Number column만 분리해서 새로운 df로 만들어준다.
df$kyphosis = ifelse(df$kyphosis == 1, "absent", "present") # df$kyphosis == 1이면 absent, 아니면 present로 replace한다.
df = mutate(df, cluster=kc$cluster) # df에 kc$cluster column을 붙인다. 이 때 새로 붙은 column의 이름은 cluster로 한다.

len = nrow(df) # df의 row의 수를 저장
for (i in 1:len) {
    print(df[i,]) # df의 i번째 row를 출력
}
