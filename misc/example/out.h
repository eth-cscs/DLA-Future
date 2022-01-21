#include <chrono>
#include <sstream>

#include <pika/runtime.hpp>

std::string out(const char* s, int i, int j, int index) {
  std::stringstream ss;
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  ss << std::put_time(&tm, "%T: ") << "t" << pika::get_worker_thread_num() << " T" << index << " " << s
     << " " << i << " " << j << std::endl;
  return ss.str();
}

std::string out(const char* s, const char* r, int index) {
  std::stringstream ss;
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  ss << std::put_time(&tm, "%T: ") << "t" << pika::get_worker_thread_num() << " T" << index << " " << r
     << " " << s << std::endl;
  return ss.str();
}
