#include "my_mpi.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "matrix.h"

// Variant 23
namespace variant {
Rectangle<double> region(Segment<double>(0, 2), Segment<double>(0, 2));
double eps = 1e-4;

double F(DoublePoint p) {
  double x2 = p.x * p.x, y2 = p.y * p.y;
  return 2 * (x2 + y2) * (1 - 2 * x2 * y2) * exp(1 - x2 * y2);
}

double Phi(DoublePoint p) {
  double x2 = p.x * p.x, y2 = p.y * p.y;
  return exp(1 - x2 * y2);
}
}  // namespace variant

// Math
namespace math {
double Laplas(const Matrix &a, const Grid &grid, IntPoint i) {
  double ldx = (a[i.x][i.y] - a[i.x - 1][i.y]) / grid.x.step();
  double rdx = (a[i.x + 1][i.y] - a[i.x][i.y]) / grid.x.step();
  double ldy = (a[i.x][i.y] - a[i.x][i.y - 1]) / grid.y.step();
  double rdy = (a[i.x][i.y + 1] - a[i.x][i.y]) / grid.y.step();
  return (ldx - rdx) / grid.x.step() + (ldy - rdy) / grid.y.step();
}

void R(const Matrix &p, const Grid &grid, Matrix *r) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int x = 1; x < r->size_x() - 1; ++x) {
    for (int y = 1; y < r->size_y() - 1; ++y) {
      IntPoint i(x, y);
      (*r)[i] = Laplas(p, grid, i) - variant::F(grid.Map(i));
    }
  }
}

void G(const Matrix &r, double alpha, Matrix *g) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int x = 1; x < r.size_x() - 1; ++x) {
    for (int y = 1; y < r.size_y() - 1; ++y) {
      IntPoint i(x, y);
      (*g)[i] = r[i] - alpha * (*g)[i];
    }
  }
}

double P(const Matrix &g, double tau, Matrix *p) {
  double diff_sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : diff_sum)
#endif
  for (int x = 1; x < p->size_x() - 1; ++x) {
    for (int y = 1; y < p->size_y() - 1; ++y) {
      IntPoint i(x, y);
      double diff = tau * g[i];
      diff_sum += diff * diff;
      (*p)[i] -= diff;
    }
  }
  return diff_sum;
}

double Alpha(const Matrix &r, const Matrix &g, const Grid &grid,
             double *ab = nullptr) {
  double a = 0, b = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : a, b)
#endif
  for (int x = 1; x < r.size_x() - 1; ++x) {
    for (int y = 1; y < r.size_y() - 1; ++y) {
      IntPoint i(x, y);
      double h = g[i] /* * grid.x.step() * grid.y.step() */;
      a += Laplas(r, grid, i) * h;
      b += Laplas(g, grid, i) * h;
    }
  }
  if (ab) {
    ab[0] = a;
    ab[1] = b;
  }
  return a / b;
}

double Tau(const Matrix &r, const Matrix &g, const Grid &grid,
           double *ab = nullptr) {
  double a = 0, b = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : a, b)
#endif
  for (int x = 1; x < r.size_x() - 1; ++x) {
    for (int y = 1; y < r.size_y() - 1; ++y) {
      IntPoint i(x, y);
      double h = g[i] /* * grid.x.step() * grid.y.step() */;
      a += r[i] * h;
      b += Laplas(g, grid, i) * h;
    }
  }
  if (ab) {
    ab[0] = a;
    ab[1] = b;
  }
  return a / b;
}
}  // namespace math

// Implementation
namespace {
int string_to_int(const std::string &number) {
  std::stringstream ss(number);
  int x;
  if (ss >> x) {
    return x;
  } else {
    throw std::logic_error("Bad number: " + number);
  }
}

int mylog2(int x) {
  int ret = 0;
  while (x) {
    x /= 2;
    ++ret;
  }
  return ret;
}

double sqr(double x) { return x * x; }

}  // namespace

double CheckAnswer(const Matrix &p, const Grid &grid) {
  double error = 0;
  for (int x = 1; x < p.size_x() - 1; ++x) {
    for (int y = 1; y < p.size_y() - 1; ++y) {
      IntPoint i(x, y);
      error += sqr(variant::Phi(grid.Map(i)) - p[i]);
    }
  }
  return error;
}

void NotParallel(Point<int> size) {
  Grid grid(Partition(variant::region.x, size.x),
            Partition(variant::region.y, size.y));

  // 0
  Matrix p(size.x, size.y), r(size.x, size.y);

  for (int x = 0; x < size.x; ++x) {
    p[x][0] = variant::Phi(grid.Map(IntPoint(x, 0)));
    p[x][size.y - 1] = variant::Phi(grid.Map(IntPoint(x, size.y - 1)));
  }
  for (int y = 0; y < size.y; ++y) {
    p[0][y] = variant::Phi(grid.Map(IntPoint(0, y)));
    p[size.x - 1][y] = variant::Phi(grid.Map(IntPoint(size.x - 1, y)));
  }

  // 1
  math::R(p, grid, &r);
  double tau = math::Tau(r, r, grid);
  math::P(r, tau, &p);

  Matrix g = r;
  // 2..N
  double difference = 1e100;
  int iter = 0;
  while (difference > variant::eps) {
    math::R(p, grid, &r);
    double alpha = math::Alpha(r, g, grid);
    math::G(r, alpha, &g);
    double tau = math::Tau(r, g, grid);
    difference = sqrt(math::P(g, tau, &p) * grid.x.step() * grid.y.step());
    std::cerr << "iteration " << iter++ << " " << difference << std::endl;
  }
  std::cerr << "error="
            << sqrt(CheckAnswer(p, grid) * grid.x.step() * grid.y.step())
            << std::endl;
}

namespace parallel {
bool debug = false;
Point<int> processes(-1, -1);
Point<int> rank(-1, -1);

Partition MakePartition(Segment<double> segment, int size, int processes,
                        int rank) {
  int quot = size / processes, rem = size % processes;
  return Partition(segment, size,
                   // std::min(rank, rem) adds points captured by first rem
                   // processors.
                   // rank != 0 adds extra point for left neighbour.
                   quot * rank + std::min(rank, rem) - (rank != 0),
                   // rank < rem adds one extra point for first rem processors.
                   // rank != size - 1 and rank != 0 add extra points for
                   // neighbours.
                   quot + (rank < rem) + (rank != processes - 1) + (rank != 0));
}

struct CommDescription {
  Rectangle<int> rectangle;
  enum Type { SEND, RECEIVE } type;
  int rank;

  CommDescription(int x_begin, int x_end, int y_begin, int y_end, Type type,
                  int rank)
      : rectangle(Segment<int>(x_begin, x_end), Segment<int>(y_begin, y_end)),
        type(type),
        rank(rank) {}
};

std::vector<CommDescription> communications;

struct Communication {
  Communication(mpi::MpiRequest *request, std::vector<double> *line,
                CommDescription *comm)
      : request(request), line(line), comm(comm) {}

  mpi::MpiRequest *request;
  std::vector<double> *line;
  CommDescription *comm;
};

void MakeCommunications(Matrix *m) {
  std::vector<Communication> requests;
  for (std::vector<CommDescription>::iterator comm = communications.begin();
       comm != communications.end(); ++comm) {
    if (comm->type == comm->SEND) {
      std::vector<double> *line = new std::vector<double>;
      for (int x = comm->rectangle.x.begin; x != comm->rectangle.x.end; ++x) {
        for (int y = comm->rectangle.y.begin; y != comm->rectangle.y.end; ++y) {
          line->push_back((*m)[x][y]);
        }
      }
      requests.push_back(
          Communication(mpi::ISend(line, comm->rank), line, &*comm));
    } else {
      std::vector<double> *line =
          new std::vector<double>(comm->rectangle.area());
      requests.push_back(
          Communication(mpi::IRecv(line, comm->rank), line, &*comm));
    }
  }
  for (std::vector<Communication>::iterator req = requests.begin();
       req != requests.end(); ++req) {
    req->request->WaitAndDelete();
    if (req->comm->type == CommDescription::RECEIVE) {
      int i = 0;
      for (int x = req->comm->rectangle.x.begin;
           x != req->comm->rectangle.x.end; ++x) {
        for (int y = req->comm->rectangle.y.begin;
             y != req->comm->rectangle.y.end; ++y) {
          (*m)[x][y] = (*req->line)[i++];
        }
      }
    }
    delete req->line;
  }
}

int MakeRankId(int x, int y) { return y * processes.x + x; }

void Parallel(Point<int> size) {
  if (const char *debug_node = getenv("DEBUG_NODE")) {
    debug = mpi::GetRank() == string_to_int(debug_node);
  }
  if (debug) {
    std::cerr << "Debugging MPI rank=" << mpi::GetRank() << std::endl;
  }
  processes.x = 1 << (mylog2(mpi::GetSize()) / 2);
  processes.y = mpi::GetSize() / processes.x;
  rank.x = mpi::GetRank() % processes.x;
  rank.y = mpi::GetRank() / processes.x;
  if (debug)
    std::cerr << "processes=" << processes << " rank=" << rank << std::endl;
  Grid grid(MakePartition(variant::region.x, size.x, processes.x, rank.x),
            MakePartition(variant::region.y, size.y, processes.y, rank.y));

  size.x = grid.x.count();
  size.y = grid.y.count();
  if (debug)
    std::cerr << " x.count=" << size.x << " x.offset=" << grid.x.offset()
              << " y.count=" << size.y << " y.offset=" << grid.y.offset()
              << std::endl;

  // Send/Receive picture for 6x1 grid with 2 nodes:
  //        0 1 2 3 4 5
  // node0  x x S R
  // node1      R S x x
  //            0 1 2 3

  if (rank.x != 0) {
    int r = MakeRankId(rank.x - 1, rank.y);
    communications.push_back(
        CommDescription(0, 1, 1, size.y - 1, CommDescription::RECEIVE, r));
    communications.push_back(
        CommDescription(1, 2, 1, size.y - 1, CommDescription::SEND, r));
  }
  if (rank.y != 0) {
    int r = MakeRankId(rank.x, rank.y - 1);
    communications.push_back(
        CommDescription(1, size.x - 1, 0, 1, CommDescription::RECEIVE, r));
    communications.push_back(
        CommDescription(1, size.x - 1, 1, 2, CommDescription::SEND, r));
  }
  if (rank.x != processes.x - 1) {
    int r = MakeRankId(rank.x + 1, rank.y);
    communications.push_back(CommDescription(
        size.x - 2, size.x - 1, 1, size.y - 1, CommDescription::SEND, r));
    communications.push_back(CommDescription(size.x - 1, size.x, 1, size.y - 1,
                                             CommDescription::RECEIVE, r));
  }
  if (rank.y != processes.y - 1) {
    int r = MakeRankId(rank.x, rank.y + 1);
    communications.push_back(CommDescription(
        1, size.x - 1, size.y - 2, size.y - 1, CommDescription::SEND, r));
    communications.push_back(CommDescription(1, size.x - 1, size.y - 1, size.y,
                                             CommDescription::RECEIVE, r));
  }

  // 0
  Matrix p(size.x, size.y), r(size.x, size.y);

  for (int x = 0; x < size.x; ++x) {
    if (rank.y == 0) p[x][0] = variant::Phi(grid.Map(IntPoint(x, 0)));
    if (rank.y == processes.y - 1)
      p[x][size.y - 1] = variant::Phi(grid.Map(IntPoint(x, size.y - 1)));
  }
  for (int y = 0; y < size.y; ++y) {
    if (rank.x == 0) p[0][y] = variant::Phi(grid.Map(IntPoint(0, y)));
    if (rank.x == processes.x - 1)
      p[size.x - 1][y] = variant::Phi(grid.Map(IntPoint(size.x - 1, y)));
  }

  // 1
  std::vector<double> ab(2), diff(1);

  math::R(p, grid, &r);
  MakeCommunications(&r);

  math::Tau(r, r, grid, ab.data());
  mpi::ReduceSum(&ab);
  double tau = ab[0] / ab[1];

  math::P(r, tau, &p);
  MakeCommunications(&p);

  Matrix g = r;
  // 2..N
  double difference = 1e100;
  int iter = 0;
  while (difference > variant::eps) {
    math::R(p, grid, &r);
    MakeCommunications(&r);

    math::Alpha(r, g, grid, ab.data());
    mpi::ReduceSum(&ab);
    double alpha = ab[0] / ab[1];

    math::G(r, alpha, &g);
    MakeCommunications(&g);

    math::Tau(r, g, grid, ab.data());
    mpi::ReduceSum(&ab);
    double tau = ab[0] / ab[1];

    diff[0] = math::P(g, tau, &p);
    MakeCommunications(&p);
    mpi::ReduceSum(&diff);
    difference = sqrt(diff[0] * grid.x.step() * grid.y.step());
    if (debug)
      std::cerr << "iteration " << iter++ << " " << difference << std::endl;
  }
  diff[0] = CheckAnswer(p, grid);
  mpi::ReduceSum(&diff);
  if (debug)
    std::cerr << "error=" << sqrt(diff[0] * grid.x.step() * grid.y.step())
              << std::endl;
}

}  // namespace parallel

int main(int ac, char **av) {
  try {
    mpi::Init(&ac, &av);

    if (ac < 3) {
      throw std::runtime_error("Usage: main size_x size_y output.txt");
    }
    Point<int> size(string_to_int(av[1]), string_to_int(av[2]));
    std::string output_filename = ac == 3 ? "" : av[3];

    double start_time = mpi::Wtime();

    if (mpi::GetSize() == 1) {
      NotParallel(size);
    } else {
      parallel::Parallel(size);
    }

    double end_time = mpi::Wtime();

    if (mpi::GetRank() == 0) {
      std::cerr << "Spent wall time: " << (end_time - start_time) << std::endl;
    }

    mpi::Finalize();
  } catch (std::exception &e) {
    std::cerr << "Unhandled exception" << std::endl << e.what() << std::endl;
    try {
      mpi::Abort();
    } catch (...) {
    }
    return 1;
  }
  return 0;
}
