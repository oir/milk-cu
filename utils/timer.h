#ifndef MILK_UTILS_TIMER_H
#define MILK_UTILS_TIMER_H

#include <chrono>
#include <unordered_map>

namespace milk {

namespace timer {

std::unordered_map<std::string,std::chrono::time_point<std::chrono::system_clock>> tic_starts;
std::unordered_map<std::string,long double> total;

void tic(std::string name = "") {
  tic_starts[name] = std::chrono::system_clock::now();
}

long double toc(std::string name = "") {
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
    (std::chrono::system_clock::now() - tic_starts[name]);
  total[name] += duration.count();
  return duration.count();
}

void print(std::string type = "mins") {
  uint max_length = 0;
  for (auto& p : timer::total)
    if (p.first.size() > max_length)
      max_length = p.first.size();
  for (auto& p : timer::total) {
    std::cout << p.first << " ";
    for (uint i=0; i<max_length-p.first.size(); i++) std::cout << "_";
    if (type == "mins")
      std::cout << "_ " << p.second / (1000*60) << " mins" << std::endl;
    else if (type == "secs")
      std::cout << "_ " << p.second / (60) << " secs" << std::endl;
    else if (type == "millis")
      std::cout << "_ " << p.second << " millis" << std::endl;
    else
      assert(false);
  }
}

const std::string date_time(const std::string format = "%Y-%m-%d_%H.%M.%S") { // is chronos overkill for this?
  std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char buff[50];
  strftime(buff, 50, format.c_str(), localtime(&tt));
  return buff;
}

} // end namespace timer

} // end namespace milk

#endif
