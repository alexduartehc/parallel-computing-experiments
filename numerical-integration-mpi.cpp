#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

int flag = 0;    	
double integral = 0; 
double total = 0;   
double f(double x);   
double Simpson(double local_a, int local_n, double h);  

main(int argc, char** argv)
{
	int my_rank;
	int p;
	double a;
	double b;
	int n;
	if(argc == 5){
		n = atoi(argv[1]);
		a = atof(argv[2]);
		b = atof(argv[3]);
		flag = atoi(argv[4]);
	}
	else{
		n = 1024;
		a = 0;
		b = 3.141592;
	}

	double h = (b-a)/n; 
	double local_a;   
	int local_n;      
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	
	double startWTime;
	if(my_rank ==0)
	{
		startWTime = MPI_Wtime();
	}
	local_n = n/p;   				 	
	local_a = a+my_rank*(b-a)/p;		
	integral = Simpson(local_a,local_n,h);  
	
	MPI_Reduce(&integral, &total, 1, MPI_DOUBLE, MPI_SUM, 0 , MPI_COMM_WORLD);
	
	double endWTime;
	if(my_rank ==0){
		endWTime = MPI_Wtime();
		cout<<"With n == "<<n<<" intervals, our estimate"<<endl;
		cout<<"of the integral from "<<a<<" to "<<b<<" is "<<total<<endl;
		cout<<"Number of processors used = "<<p<<endl;
		cout<<"Time elapsed: "<<(endWTime-startWTime)*1000<<"ms"<<endl;
		fflush(stdout); 
	}

	MPI_Finalize();
}


double Simpson(double local_a, int local_n, double h){
	double result = 0;
	for(int i =0; i<=local_n; i++)
	{
		if((i==0)||(i==local_n))
			result += f(local_a + i* h);
		else if(i%2==0)

			result += 2* f(local_a + i*h);
		else
			result += 4* f(local_a + i*h);	
	}
	result *= h/3;  
	return(result);
}

double f(double x){
	double return_val;
	switch (flag) 
	{
    case 0:
      return_val = sin(x);
      break;
    case 1:
      return_val = cos(x);
      break;   
    case 2:
      return_val = tan(x);
      break;
    case 3:
      return_val = 1/x;
      break;   
    default:
      return_val = cos(x);
      break;
  }
	return return_val;
}
