#ifndef MPI_H_
#define MPI_H_

#include <mpi.h>
#include <vector>

namespace mpi {

#define MYMPI_CHECKED(X) mpi::Check((X), #X, __FILE__, __LINE__)
int Check(int rc, const char *fn, const char *file, int line);

void Init(int *ac, char ***av);

void Finalize();

void Abort();

void Barrier();

// Also calls Barrier().
double Wtime();

int GetRank();

int GetSize();

class MpiRequest {
 public:
  void WaitAndDelete();

 private:
  MpiRequest();
  ~MpiRequest();
  MPI_Request request_;

  friend MpiRequest *ISend(const std::vector<double> *data, int dest);
  friend MpiRequest *IRecv(std::vector<double> *data, int src);
};

MpiRequest *ISend(const std::vector<double> *data, int dest);
MpiRequest *IRecv(std::vector<double> *data, int src);

void ReduceSum(std::vector<double> *data);

}  // namespace mpi

#endif
