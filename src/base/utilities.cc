#include <felspa/base/utilities.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace util
/* -------------------------------------------*/
{
  std::string get_file_extension(const std::string& filename)
  {
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash == std::string::npos) last_slash = 0;

    size_t dot_pos = filename.rfind('.');
    if (dot_pos <= last_slash || dot_pos == std::string::npos) return "";

    std::string extension = filename.substr(dot_pos + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return extension;
  }


  std::string demangle_typeid(const char* typeid_name)
  {
    int status = -4;

    // by default __cxa_demangle will dump the result on the heap
    // that part of the memory must be released by std::free
    std::unique_ptr<char, void (*)(void*)> demangled_name{
      abi::__cxa_demangle(typeid_name, nullptr, nullptr, &status), std::free};

    // if somehow demangle fails, return the un-demangled name
    return (status == 0) ? demangled_name.get() : typeid_name;
  }


  std::string get_date_time(const std::string& format_string)
  {
    auto now_time = std::chrono::system_clock::now();
    auto now_in_time_t = std::chrono::system_clock::to_time_t(now_time);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_in_time_t), &format_string[0]);
    return ss.str();
  }
}  // namespace util

FELSPA_NAMESPACE_CLOSE
