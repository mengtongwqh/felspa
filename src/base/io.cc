#include <deal.II/base/data_out_base.h>
#include <felspa/base/io.h>
#include <felspa/base/log.h>
#include <felspa/base/types.h>
#include <felspa/base/utilities.h>


FELSPA_NAMESPACE_OPEN

/* ************************************************** */
/*  class ExportFileFormatDict                        */
/* ************************************************** */
ExportFileFormat ExportFileFormatDict::operator[](
  const std::string& file_extension)
{
  try {
    return string_format_dict.at(file_extension);
  }
  catch (std::out_of_range& e) {
    THROW(ExcNotInFileFormatDict());
  }
  catch (...) {
    THROW(ExcInternalErr());
  }
}


// instantiate one as global object
ExportFileFormatDict export_format_dict;


std::ostream& operator<<(std::ostream& os, ExportFileFormat fmt)
{
  std::string str;

#define PROCESS(val)          \
  case ExportFileFormat::val: \
    str = #val;               \
    break

  switch (fmt) {
    PROCESS(gnuplot);
    PROCESS(pvd);
    PROCESS(svg);
    PROCESS(vtk);
    PROCESS(vtu);
    PROCESS(npy);
    PROCESS(npz);
    default:
      THROW(ExcInternalErr());
  }

#undef PROCESS

  return os << str;
}


/* ************************************************** */
/*  class ExportFile                                  */
/* ************************************************** */
ExportFile::ExportFile(const std::string& file_name_)
  : file_name(file_name_), file_extension(util::get_file_extension(file_name_))
{
  ASSERT_NON_EMPTY(file_extension);

  // parse the format at set the appropriate format string
  std::string format_string;
  format = export_format_dict[file_extension];

  // open the file and attach to stream
  // if the open fails, throw exception
  stream.open(file_name);
  ASSERT(stream.is_open(), ExcFailToOpen(file_name));
}


ExportFile::~ExportFile()
{
  /*
   * first test if stream is all good
   * if not, report an error message.
   * but we don't want to throw an exception
   * because doing so will leave the stream
   * in an unclosed state
   */
  ASSERT_WARN(stream.good(),
              ExcStreamInAbnormalState(file_name, stream.eof(), stream.fail(),
                                       stream.bad()));
  /*
   * only close the stream if it is still open
   * or close() will fail
   */
  if (stream.is_open()) stream.close();
}


std::ofstream& ExportFile::access_stream()
{
  ASSERT_WARN(stream.good(),
              ExcStreamInAbnormalState(file_name, stream.eof(), stream.fail(),
                                       stream.bad()));
  return stream;
}


/* ************************************************** */
/*  class PVDCollector                                */
/* ************************************************** */
template <typename TimeStepType>
PVDCollector<TimeStepType>::PVDCollector(const std::string& file_name_)
  : file_name(file_name_)
{
  ASSERT_NON_EMPTY(file_name_);
}


template <typename TimeStepType>
PVDCollector<TimeStepType>::~PVDCollector()
{
  write_pvd_records();
}


template <typename TimeStepType>
void PVDCollector<TimeStepType>::set_file_path(const std::string& filepath)
{
  ASSERT_NON_EMPTY(filepath);
  file_path = filepath;
}


template <typename TimeStepType>
void PVDCollector<TimeStepType>::set_file_name(const std::string& filename)
{
  ASSERT_NON_EMPTY(filename);
  file_name = filename;
}


template <typename TimeStepType>
void PVDCollector<TimeStepType>::append_record(time_step_type current_time,
                                               const std::string& file_name)
{
  ASSERT(current_time >= 0, ExcInternalErr());
  ASSERT_NON_EMPTY(file_name);
  phsxtime_outfile.push_back({current_time, file_name});
}


template <typename TimeStepType>
void PVDCollector<TimeStepType>::write_pvd_records()
{
  if (phsxtime_outfile.empty()) return;

  LOG_PREFIX("PVDCollector");
  felspa_log << "Exporting pvd collection containing " << get_file_count()
             << " files...";

  std::string pvd_filename = file_path + file_name + ".pvd";
  ExportFile pvd_file(pvd_filename);
  dealii::DataOutBase::write_pvd_record(pvd_file.access_stream(),
                                        phsxtime_outfile);

  felspa_log << "Done." << std::endl;
}


/*********** Explicit Instantiations ***********/
template class PVDCollector<types::DoubleType>;
/***********************************************/

FELSPA_NAMESPACE_CLOSE
