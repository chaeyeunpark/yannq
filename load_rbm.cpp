#include <iostream>
#include <iomanip>
#include <fstream>
#include <ios>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/archive/text_oarchive.hpp>

#include <regex>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/program_options.hpp>

#include "Machines/RBM.hpp"
#include "Serializers/SerializeRBM.hpp"

#include <nlohmann/json.hpp>

#include <cnpy.h>

template<typename T, bool useBias>
void save_to_npz(const yannq::RBM<T, useBias>& qs, const boost::filesystem::path& output_file )
{
	Eigen::MatrixXcd W = qs.getW();
	cnpy::npz_save(output_file.c_str(), "W", W.data(), {W.rows(), W.cols()}, "w");

	Eigen::VectorXcd A = qs.getA();
	cnpy::npz_save(output_file.c_str(), "A", A.data(), {A.rows()}, "a");

	Eigen::VectorXcd B = qs.getB();
	cnpy::npz_save(output_file.c_str(), "B", B.data(), {B.rows()}, "a");
}

template<typename T, bool useBias>
void print(const yannq::RBM<T, useBias>& qs, std::ostream& out)
{
	boost::archive::text_oarchive oa(out);
	oa << qs;
}

template<typename T, bool useBias>
void load_qs_from_file(const boost::filesystem::path& input_file, yannq::RBM<T, useBias>& qs)
{
	using std::ios;
	using namespace boost::filesystem;
	fstream in(input_file, ios::binary | ios::in);
	{
		boost::archive::binary_iarchive ia(in);
		ia >> qs;
	}
}

template<typename T, bool useBias>
void process_dir(
		const boost::filesystem::path& input_path, 
		const boost::filesystem::path& output_path )
{
	using std::ios;
	using namespace boost::filesystem;
	using namespace boost;

	std::regex exp("^w([0-9]{4}).dat$", std::regex::extended);
	yannq::RBM<T, useBias> qs(1,1);

	for(auto& entry: make_iterator_range(directory_iterator(input_path), {}))
	{
		if(!is_regular_file(entry))
			continue;

		path filePath = entry.path();
		
		std::cmatch what;
		std::string fileName = filePath.filename().string();
		bool matched;
		matched = std::regex_match(fileName.c_str(), what, exp);
		if(!matched)
			continue;

		int w;
		std::string m = what[1].str();
		sscanf(m.c_str(), "%d", &w);

		load_qs_from_file(filePath, qs);;
		
		{
			char fileName[255];

			sprintf(fileName, "w%04d.npz", w);
			path p = output_path;
			p /= fileName;

			save_to_npz(qs, p);
		}
	}
}

int main(int argc, char** argv)
{
	using namespace boost::filesystem;
	using namespace boost::program_options;
	using nlohmann::json;

	do
	{
		options_description desc{"Options"};
		desc.add_options()
			("help,h", "Help screen")
			("dir", bool_switch()->default_value(true), "Set if input is directory")
			("file", bool_switch()->default_value(false), "Set if input is file")
			("print", bool_switch()->default_value(false), "print to stdout in text form")
			("input,i", value<std::string>()->default_value(""), "Path for input file or directory")
			("output,o", value<std::string>()->default_value(""), "Path for output file or directory");

		positional_options_description pd;
		pd.add("input", 1).add("output", 1);

		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);

		if (vm.count("help"))
		{
			std::cout << desc << '\n';
			break;
		}

		bool usedir = vm["dir"].as<bool>();
		if(vm["file"].as<bool>())
			usedir = false;
		
		std::string input = vm["input"].as<std::string>();
		std::string output = vm["output"].as<std::string>();
		//std::cout << input << "\t " << output << std::endl;
		
		path input_path = input;
		path output_path = output;

		if (input.empty())
		{
			std::cout << "You should provide input" << std::endl;
			break;
		}

		if (usedir)
		{
			if (output.empty())
			{
				std::cout << "You should provide both input and output" << std::endl;
				break;
			}
			if(!exists(output_path))
			{
				create_directory(output_path);
			}
			else if(!is_directory(output_path))
			{
				std::cout << "Output must be directory" << std::endl;
				break;
			}
			process_dir<std::complex<double>, true>(input_path, output_path);
			break;
		}
		else
		{
			yannq::RBM<std::complex<double>, true> qs(1,1);
			load_qs_from_file(input_path, qs);
			if(vm["print"].as<bool>())
			{
				print(qs, std::cout);
				break;
			}
			else if (output.empty())
			{
				std::cout << "You should provide both input and output" << std::endl;
				break;
			}
			save_to_npz<std::complex<double>, true>(qs, output_path);
			break;
		}
	}
	while(false);

	return 0;
}
