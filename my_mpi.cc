#include "my_mpi.h"

#include <sstream>

#define CHECKED(X) MYMPI_CHECKED(X)

namespace mpi {

namespace {
int rank = -1;
int size = -1;
}  // namespace

int Check(int rc, const char *fn, const char *file, int line) {
  if (rc != MPI_SUCCESS) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << fn << ": MPI error " << rc;
    throw std::runtime_error(ss.str());
  }
  return rc;
}

void Init(int *ac, char ***av) {
  CHECKED(MPI_Init(ac, av));
  CHECKED(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECKED(MPI_Comm_size(MPI_COMM_WORLD, &size));
}

void Finalize() { CHECKED(MPI_Finalize()); }

void Abort() { CHECKED(MPI_Abort(MPI_COMM_WORLD, 1)); }

void Barrier() { CHECKED(MPI_Barrier(MPI_COMM_WORLD)); }

double Wtime() {
  Barrier();
  return MPI_Wtime();
}

int GetRank() { return rank; }

int GetSize() { return size; }

MpiRequest *ISend(const std::vector<double> *data, int dest) {
  MpiRequest *ret = new MpiRequest;
  try {
    CHECKED(MPI_Isend(data->data(), data->size(), MPI_DOUBLE, dest, 0,
                      MPI_COMM_WORLD, &ret->request_));
  } catch (...) {
    delete ret;
    throw;
  }
  return ret;
}

MpiRequest *IRecv(std::vector<double> *data, int src) {
  MpiRequest *ret = new MpiRequest;
  try {
    CHECKED(MPI_Irecv(data->data(), data->size(), MPI_DOUBLE, src, 0,
                      MPI_COMM_WORLD, &ret->request_));
  } catch (...) {
    delete ret;
    throw;
  }
  return ret;
}

void MpiRequest::WaitAndDelete() {
  CHECKED(MPI_Wait(&request_, MPI_STATUS_IGNORE));
  delete this;
}

MpiRequest::MpiRequest() {}

MpiRequest::~MpiRequest() {}

void ReduceSum(std::vector<double> *data) {
  std::vector<double> tmp = *data;
  CHECKED(MPI_Allreduce(data->data(), tmp.data(), data->size(), MPI_DOUBLE,
                        MPI_SUM, MPI_COMM_WORLD));
  *data = tmp;
}

}  // namespace mpi
