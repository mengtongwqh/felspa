#ifndef _FELSPA_BASE_IO_H_
#define _FELSPA_BASE_IO_H_

#include <deal.II/dofs/dof_handler.h>
#include <felspa/base/exceptions.h>
#include <felspa/base/felspa_config.h>

#include <fstream>
#include <memory>
#include <string>

FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/**
 * A List of supported file formats
 */
/* ************************************************** */
enum class ExportFileFormat
{
  csv,      //!< csv file for output data
  gnuplot,  //!< gnuplot, for sparsity
  pvd,      //!< pvd, collection of vtk for animation
  svg,      //!< svg format, for sparsity
  vtk,      //!< vtk, for post-processing
  vtu,      //!< vtu, for post-processing
  npy,      //!< for numpy matrix visualization in Python
  npz       //!< for numpy matrix in Python, compressed
};


/// export the file format to \c ostream
std::ostream& operator<<(std::ostream& os, ExportFileFormat fmt);


/* ************************************************** */
/**
 * Convert a string to  \enum ExportFileFormat.
 */
/* ************************************************** */
extern class ExportFileFormatDict
{
 public:
  /**
   * Throw exception when file extension is not in the map
   */
  ExportFileFormat operator[](const std::string&);


  DECL_EXCEPT_0(ExcNotInFileFormatDict,
                "The input ExportFileFormat is not in the string2format table");


 protected:
  /**
   * Map of file extension string to supporteed file formats.
   * Sometimes there are multiple strings corresponding to the same format.
   */
  const std::map<std::string, ExportFileFormat> string_format_dict{
    {"csv", ExportFileFormat::csv}, {"gnuplot", ExportFileFormat::gnuplot},
    {"npy", ExportFileFormat::npy}, {"npz", ExportFileFormat::npz},
    {"pvd", ExportFileFormat::pvd}, {"svg", ExportFileFormat::svg},
    {"vtk", ExportFileFormat::vtk}, {"vtu", ExportFileFormat::vtu}};
} export_format_dict;


/* ************************************************** */
/**
 * \class ExportFile
 * This is a class to handle file output
 * designed with RAII in mind
 */
/* ************************************************** */
class ExportFile
{
 public:
  /** @name Basic Object Operations  */
  //@{
  /**
   * Constructor
   * @param file_name_[in] the path and name of the file
   * @param file_format_[in] the file format
   */
  ExportFile(const std::string& file_name);

  ExportFile(const ExportFile&) = delete;

  ExportFile& operator=(const ExportFile&) = delete;

  /**
   * Destructor, within which the file stream is closed
   */
  ~ExportFile();
  //@}


  /**
   * @name Member Getter/Accessor
   */
  //@{
  std::ofstream& access_stream();

  std::string get_file_name() const { return file_name; }

  ExportFileFormat get_format() const { return format; }

  std::string get_file_extension() const { return file_extension; }
  //@}


  /** @name Exceptions */
  //@{
  DECL_EXCEPT_1(ExcFailToOpen, "Fail to open file " << arg1, std::string);

  DECL_EXCEPT_4(ExcStreamInAbnormalState,
                "The stream related to file "
                  << arg1 << " is in an abnormal state, with eof = " << arg2
                  << " ,fail = " << arg3 << " ,bad = " << arg4,
                std::string, bool, bool, bool);
  //@}


 private:
  std::string file_name;


  std::string file_extension;


  ExportFileFormat format;


  std::ofstream stream;
};


/* ************************************************** */
/**
 * This is object creates a pvd collection of vtu files
 * for animation with the physical time info.
 * The creation of the pvd file is done at destruction time.
 */
/* ************************************************** */
template <typename TimeStepType>
class PVDCollector
{
 public:
  using time_step_type = TimeStepType;
  using size_type = typename std::vector<time_step_type>::size_type;

  /**
   * Constructor.
   */
  PVDCollector(const std::string& file_name_);


  /**
   * Destructor.
   */
  ~PVDCollector();


  /**
   * The path to write the pvd file.
   */
  void set_file_path(const std::string& filepath);


  /**
   * Set the name of the pvd file.
   */
  void set_file_name(const std::string& filename);


  /**
   * @brief Get file name
   */
  const std::string& get_file_name() const { return file_name; }


  /**
   * @brief Get file path
   */
  const std::string& get_file_path() const { return file_path; }


  /**
   * Clear all existing records. Keep the filename and filepath
   */
  void clear() { phsxtime_outfile.clear(); }


  /**
   * Add the current time step.
   */
  void append_record(time_step_type current_time, const std::string& file_name);


  /**
   * Obtain the file count.
   */
  size_type get_file_count() const { return phsxtime_outfile.size(); }


 protected:
  /**
   * Filename without extension or path.
   */
  std::string file_name;


  /**
   * Path of output *.pvd file.
   */
  std::string file_path = "./";


  /**
   * A vector that stores a pair containing the current physical time and
   * the name of the vtu file name.
   */
  std::vector<std::pair<time_step_type, std::string>> phsxtime_outfile;


 private:
  /**
   * This will be called when the object is destructed to spit out
   *
   */
  void write_pvd_records();
};


FELSPA_NAMESPACE_CLOSE
#endif /* _FELSPA_BASE_IO_H_ */
