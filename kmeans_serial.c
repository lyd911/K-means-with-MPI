 /*Serial code for kmeans algorithm*/
 
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <time.h>
 
 int K,N,D;  // cluster numbers, data size, dimension
 float **data;  //  data storage
 int *in_cluster;  //  to indicate which cluster every point belongs to
 float **cluster_center;  //  to storage cluster center
 
 float **array(int m,int n);
 void freearray(float **p);
 float **loadData(int *k,int *d,int *n);
 float getDistance(float avector[],float bvector[],int n);
 void cluster();
 float getTotalDistance();
 void getCenter(int in_cluster[N]);
 
 int  main()
 {
     int i,j,count=0;
     float temp1,temp2;
     data=loadData(&K,&D,&N);
     printf("Data sets:n");
     for(i=0;i<N;i++)
         for(j=0;j<D;j++){
             printf("%-8.2f",data[i][j]);
             if((j+1)%D==0)    putchar('n');
         }
     printf("-----------------------------n");
 
     srand((unsigned int)(time(NULL)));  //randomly initialize k cluster center
     for(i=0;i<K;i++)
         for(j=0;j<D;j++)
             cluster_center[i][j]=data[(int)(N*rand()/(RAND_MAX+1.0))][j];
 
     cluster();  //cluster with k initial centers
     temp1=getTotalDistance();  //get the total distance of the cluster center and the data points for the first time
     count++;
     printf("The difference between data and center is: %.2fnn", temp1);
 
     getCenter(in_cluster);
     cluster();  //cluster with k new centers
     temp2=getTotalDistance();
     count++;
     printf("The difference between data and center is: %.2fnn",temp2);
 
     while(fabs(temp2-temp1)!=0){   //compare between the two iterations, if not the same, continue iteration
         temp1=temp2;
         getCenter(in_cluster);
         cluster();
         temp2=getTotalDistance();
         count++;
         printf("The %dth difference between data and center is: %.2fnn",count,temp2);
     }
 
     printf("The total number of cluster is: %dnn",count);  //calculate the number of iterations
     system("pause");
     return 0;
 }
 
 
 //dynamically declare an array
 float **array(int m,int n)
 {
     float **p;
     p=(float **)malloc(m*sizeof(float *));
     p[0]=(float *)malloc(m*n*sizeof(float));
     for(int i=1;i<m;i++)    p[i]=p[i-1]+n;
     return p;
 }
 
 //free the memory allocated to the array
 void freearray(float **p)
 {
     free(*p);
     free(p);
 }
 
 //import data from "data.txt", format for first line: K=number of clusters, D=dimension, N=data size
 float **loadData(int *k,int *d,int *n)
 {
     float **arraydata;
     FILE *fp;
     if((fp=fopen("data.txt","r"))==NULL)    fprintf(stderr,"cannot open data.txt!n");
     if(fscanf(fp,"K=%d,D=%d,N=%dn",k,d,n)!=3)        fprintf(stderr,"load error!n");
     arraydata=array(*n,D);  //initialize the data set
     cluster_center=array(*k,D);  //get the cluster center
     in_cluster=(int *)malloc(*n * sizeof(int));  //an array to store the clusters every data point belongs to
     for(int i=0;i<*n;i++)
         for(int j=0;j<D;j++)
             fscanf(fp,"%f",&arraydata[i][j]);  //read from the data point
     return arraydata;
 }
 
 //calculate the Euclidean distance
 float getDistance(float avector[],float bvector[],int n)
 {
     int i;
     float sum=0.0;
     for(i=0;i<n;i++)
         sum+=pow(avector[i]-bvector[i],2);
     return sqrt(sum);
 }
 
 //cluster method for N data points, mark which cluster every point belongs to
 void cluster()
 {
     int i,j;
     float min;
     float **distance=array(N,K);  //store the distance between every point to every cluster center
     //float distance[N][K];  
     for(i=0;i<N;++i){
         min=9999.0;
         for(j=0;j<K;++j){
             distance[i][j] = getDistance(data[i],cluster_center[j],D);
             //printf("%fn", distance[i][j]);
             if(distance[i][j]<min){
                 min=distance[i][j];
                 in_cluster[i]=j;
             }
         }
         printf("data[%d] in cluster-%dn",i,in_cluster[i]+1);
     }
     printf("-----------------------------n");
     free(distance);
 }
 
 //calculate the total distance between all cluster centers and all data points
 float getTotalDistance()
 {
     int i,j;
     float sum=0.0;
     for(i=0;i<K;++i){
         for(j=0;j<N;++j){
             if(i==in_cluster[j])
                 sum+=getDistance(data[j],cluster_center[i],D);
         }
     }
     return sum;
 }
 
 //calculate every cluster center
 void getCenter(int in_cluster[N])
 {
     float **sum=array(K,D);  //store every cluster center
     //float sum[K][D];  
     int i,j,q,count;
     for(i=0;i<K;i++)
         for(j=0;j<D;j++)
             sum[i][j]=0.0;
     for(i=0;i<K;i++){
         count=0;  //count the total number of data points in this cluster
         for(j=0;j<N;j++){
             if(i==in_cluster[j]){
                 for(q=0;q<D;q++)
                     sum[i][q]+=data[j][q];  //calculate the total dimension of all data points in this cluster
                 count++;
             }
         }
         for(q=0;q<D;q++)
             cluster_center[i][q]=sum[i][q]/count;
     }
     printf("The new center of cluster is:n");
     for(i = 0; i < K; i++)
         for(q=0;q<D;q++){
             printf("%-8.2f",cluster_center[i][q]);
             if((q+1)%D==0)    putchar('n');
     }
     free(sum);
 }
