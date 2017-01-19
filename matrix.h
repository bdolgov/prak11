#ifndef MARTIX_H
#define MARTIX_H

#include <iostream>
#include <vector>

template <class T>
struct Point {
  Point(T x, T y) : x(x), y(y) {}

  T x;
  T y;
};

template <class T>
std::ostream &operator<<(std::ostream &out, const Point<T> &p) {
  return out << "(" << p.x << ", " << p.y << ")";
}

typedef Point<int> IntPoint;
typedef Point<double> DoublePoint;

class Matrix {
 public:
  Matrix(int size_x, int size_y)
      : size_x_(size_x), size_y_(size_y), content_(size_x_ * size_y_, 0.0) {}

  double *operator[](int idx) { return content_.data() + size_y_ * idx; }

  const double *operator[](int idx) const {
    return content_.data() + size_y_ * idx;
  }

  double &operator[](IntPoint p) { return operator[](p.x)[p.y]; }

  double operator[](IntPoint p) const { return operator[](p.x)[p.y]; }

  int size_x() const { return size_x_; }
  int size_y() const { return size_y_; }

 private:
  int size_x_, size_y_;
  std::vector<double> content_;
};

template <class T>
struct Segment {
  Segment(T begin, T end) : begin(begin), end(end) {}

  T begin;
  T end;
};

template <class T>
struct Rectangle {
  Rectangle(Segment<T> x, Segment<T> y) : x(x), y(y) {}

  Segment<T> x;
  Segment<T> y;

  T area() const { return (x.end - x.begin) * (y.end - y.begin); }
};

class Partition {
 public:
  Partition(Segment<double> segment, int count, int offset = 0,
            int real_count = -1)
      : begin_(segment.begin),
        step_((segment.end - segment.begin) / count),
        offset_(offset),
        count_(real_count != -1 ? real_count : count) {}

  double Map(int idx) const { return begin_ + step_ * (idx + offset_); }

  double step() const { return step_; }

  int count() const { return count_; }

  int offset() const { return offset_; }

 private:
  const double begin_, step_;
  int offset_, count_;
};

struct Grid {
 public:
  Grid(Partition x, Partition y) : x(x), y(y) {}

  Point<double> Map(Point<int> p) const {
    return Point<double>(x.Map(p.x), y.Map(p.y));
  }

  const Partition x, y;
};

#endif
