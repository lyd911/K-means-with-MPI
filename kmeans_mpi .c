/*mpi version of kmeans algorithm*/

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <time.h>
 #include "mpi.h"
 
 int  main(int argc,char *argv[])
 {
     int i,j;
     MPI_Status status;
     float temp1,temp2;
     int K,N,D;  // cluster numbers, data size, dimension
     float **data;  // data storage
     int *all_in_cluster;  //rank 0 to indicate which cluster every point belongs to
     int *local_in_cluster;  //other ranks mark which cluster this data point belongs to
     int *in_cluster;  //rank 0 to indicate which cluster every point belongs to
     int count=0;
     float *sum_diff;
     float *global_sum_diff;
     float **cluster_center;  //to storage cluster center
     int rank,size;
     float **array(int m,int n);
     float **loadData(int *k,int *d,int *n);
     float getDistance(float avector[],float bvector[],int n);
     void cluster(int n,int k,int d,float **data,float **cluster_center,int *local_in_cluster);
     float getDifference(int k,int n,int d,int *in_cluster,float **data,float **cluster_center,float *sum);
     void getCenter(int k,int d,int n,int *in_cluster,float **data,float **cluster_center);
 
     MPI_Init(&argc,&argv);
     MPI_Comm_rank(MPI_COMM_WORLD,&rank);
     MPI_Comm_size(MPI_COMM_WORLD,&size);
     if(!rank){
         data=loadData(&K,&D,&N);  //rank 0 load data
         if(size==1||size>N||N%(size-1))    MPI_Abort(MPI_COMM_WORLD,1);  //quit if not readable
     }
     MPI_Bcast(&K,1,MPI_INT,0,MPI_COMM_WORLD);  //rank 0 broadcast data to other ranks
     MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);  
     MPI_Bcast(&D,1,MPI_INT,0,MPI_COMM_WORLD);  
     if(rank)    data=array(N/(size-1),D);  //other ranks instantiate memory lacation
     all_in_cluster=(int *)malloc(N/(size-1)*size*sizeof(int));  //for rank 0
     local_in_cluster=(int *)malloc(N/(size-1)*sizeof(int));  //for every rank
     in_cluster=(int *)malloc(N*sizeof(int));  //for rank 0
     sum_diff=(float *)malloc(K*sizeof(float));  //the total distance of all data points to cluster centers in this cluster for every rank
     global_sum_diff=(float *)malloc(K*sizeof(float));
     for(i=0;i<K;i++)    sum_diff[i]=0.0;  //Initialize
 
     if(!rank){  //rank 0 assign work to other ranks
         for(i=0;i<N;i+=(N/(size-1)))
             for(j=0;j<(N/(size-1));j++)
                 MPI_Send(data[i+j],D,MPI_FLOAT,(i+j)/(N/(size-1))+1,99,MPI_COMM_WORLD);  
         printf("Data sets:n");
         for(i=0;i<N;i++)
             for(j=0;j<D;j++){
                 printf("%-8.2f",data[i][j]);
                 if((j+1)%D==0)    putchar('n');
             }
            printf("-----------------------------n");
     }else{  //other ranks receive work assignment
         for(i=0;i<(N/(size-1));i++)
             MPI_Recv(data[i],D,MPI_FLOAT,0,99,MPI_COMM_WORLD,&status);
     }
     MPI_Barrier(MPI_COMM_WORLD);  //sync
     cluster_center=array(K,D);  //cluster center
     if(!rank){  //rank 0 randomly initialize centers
         srand((unsigned int)(time(NULL)));  //randomly initialize k centers
         for(i=0;i<K;i++)
             for(j=0;j<D;j++)
                 cluster_center[i][j]=data[(int)((double)N*rand()/(RAND_MAX+1.0))][j];
     }
     for(i=0;i<K;i++)    MPI_Bcast(cluster_center[i],D,MPI_FLOAT,0,MPI_COMM_WORLD);  //rank 0 broadcast centers to other ranks
     if(rank){
         cluster(N/(size-1),K,D,data,cluster_center,local_in_cluster);  //cluster for other ranks
         getDifference(K,N/(size-1),D,local_in_cluster,data,cluster_center,sum_diff);
         for(i=0;i<N/(size-1);i++)
             printf("data[%d] in cluster-%dn",(rank-1)*(N/(size-1))+i,local_in_cluster[i]+1);
     }
     MPI_Gather(local_in_cluster,N/(size-1),MPI_INT,all_in_cluster,N/(size-1),MPI_INT,0,MPI_COMM_WORLD);  //gather data to rank 0
     MPI_Reduce(sum_diff,global_sum_diff,K,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);  //gather to rank 0, get the total distance of the cluster centers and the data points in one cluster
     if(!rank){  
         for(i=N/(size-1);i<N+N/(size-1);i++) 
             in_cluster[i-N/(size-1)]=all_in_cluster[i];  //handle the array
         temp1=0.0;
         for(i=0;i<K;i++) temp1+=global_sum_diff[i];
         printf("The difference between data and center is: %.2fnn", temp1);
         count++;
     }
     MPI_Bcast(&temp1,1,MPI_FLOAT,0,MPI_COMM_WORLD);
     MPI_Barrier(MPI_COMM_WORLD);
 
     do{   //compare between the two iterations, if not the same, continue iteration
         temp1=temp2;
         if(!rank)    getCenter(K,D,N,in_cluster,data,cluster_center);  //re-calculte the centers
         for(i=0;i<K;i++)    MPI_Bcast(cluster_center[i],D,MPI_FLOAT,0,MPI_COMM_WORLD);  //broadcast the new centers   
         if(rank){
             cluster(N/(size-1),K,D,data,cluster_center,local_in_cluster);  //cluster for other ranks
             for(i=0;i<K;i++)    sum_diff[i]=0.0;
             getDifference(K,N/(size-1),D,local_in_cluster,data,cluster_center,sum_diff);
             for(i=0;i<N/(size-1);i++)
                 printf("data[%d] in cluster-%dn",(rank-1)*(N/(size-1))+i,local_in_cluster[i]+1);
         }
         MPI_Gather(local_in_cluster,N/(size-1),MPI_INT,all_in_cluster,N/(size-1),MPI_INT,0,MPI_COMM_WORLD);
         if(!rank)
             for(i=0;i<K;i++)    global_sum_diff[i]=0.0;
         MPI_Reduce(sum_diff,global_sum_diff,K,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
         if(!rank){
             for(i=N/(size-1);i<N+N/(size-1);i++) 
                 in_cluster[i-N/(size-1)]=all_in_cluster[i];
             temp2=0.0;
             for(i=0;i<K;i++) temp2+=global_sum_diff[i];
             printf("The difference between data and center is: %.2fnn", temp2);
             count++;
         }
         MPI_Bcast(&temp2,1,MPI_FLOAT,0,MPI_COMM_WORLD);
         MPI_Barrier(MPI_COMM_WORLD);
     }while(fabs(temp2-temp1)!=0.0);
     if(!rank)    printf("The total number of cluster is: %dnn",count);
     MPI_Finalize();
 }
 
 
 //dynamically declare an array
 float **array(int m,int n)
 {
     int i;
     float **p;
     p=(float **)malloc(m*sizeof(float *));
     p[0]=(float *)malloc(m*n*sizeof(float));
     for(i=1;i<m;i++)    p[i]=p[i-1]+n;
     return p;
 }
 
 //import data from "data.txt", format for first line: K=number of clusters, D=dimension, N=data size
 float **loadData(int *k,int *d,int *n)
 {
     float **array(int m,int n);
     int i,j;
     float **arraydata;
     FILE *fp;
     if((fp=fopen("data.txt","r"))==NULL)    fprintf(stderr,"cannot open data.txt!n");
     if(fscanf(fp,"K=%d,D=%d,N=%dn",k,d,n)!=3)    fprintf(stderr,"load error!n");
     arraydata=array(*n,*d);  //initialize the data set
     for(i=0;i<*n;i++)
         for(j=0;j<*d;j++)
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
 void cluster(int n,int k,int d,float **data,float **cluster_center,int *local_in_cluster)
 {
     int i,j;
     float min;
     float **distance=array(n,k);  //store the distance between every point to every cluster center
     for(i=0;i<n;++i){
         min=9999.0;
         for(j=0;j<k;++j){
             distance[i][j] = getDistance(data[i],cluster_center[j],d);
             if(distance[i][j]<min){
             min=distance[i][j];
             local_in_cluster[i]=j;
         }
        }
     }
     printf("-----------------------------n");
     free(distance);
 }
 
 //calculate the total distance between all cluster centers and all data points
 float getDifference(int k,int n,int d,int *in_cluster,float **data,float **cluster_center,float *sum)
 {
     int i,j;
     for(i=0;i<k;++i)
         for(j=0;j<n;++j)
             if(i==in_cluster[j])
                 sum[i]+=getDistance(data[j],cluster_center[i],d);
 }
 
 //calculate every cluster center
 void getCenter(int k,int d,int n,int *in_cluster,float **data,float **cluster_center)
 {
     float **sum=array(k,d);  //store every cluster center
     int i,j,q,count;
     for(i=0;i<k;i++)
         for(j=0;j<d;j++)
             sum[i][j]=0.0;
     for(i=0;i<k;i++){
         count=0;  //count the total number of data points in this cluster
         for(j=0;j<n;j++){
             if(i==in_cluster[j]){
                 for(q=0;q<d;q++)
                     sum[i][q]+=data[j][q];  //calculate the total dimension of all data points in this cluster
                 count++;
             }
         }
         for(q=0;q<d;q++)
             cluster_center[i][q]=sum[i][q]/count;
     }
     printf("The new center of cluster is:n");
         for(i = 0; i < k; i++)
             for(q=0;q<d;q++){
                 printf("%-8.2f",cluster_center[i][q]);
                 if((q+1)%d==0)    putchar('n');
     }
     free(sum);
 }