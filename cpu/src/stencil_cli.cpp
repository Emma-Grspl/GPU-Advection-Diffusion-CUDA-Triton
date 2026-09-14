#include "stencil/AdvectionDiffusion.hpp"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


namespace
{

void read_input_csv(
    const std::string& path,
    std::vector<double>& phi,
    std::vector<double>& u,
    std::vector<double>& v
)
{
    std::ifstream input(path);

    if (!input)
    {
        throw std::runtime_error(
            "Cannot open input file: "
            + path
        );
    }

    std::string header;

    if (!std::getline(input, header))
    {
        throw std::runtime_error(
            "Input CSV is empty."
        );
    }

    if (
        !header.empty()
        && header.back() == '\r'
    )
    {
        header.pop_back();
    }

    if (header != "k,phi,u,v")
    {
        throw std::runtime_error(
            "Unexpected CSV header: "
            + header
        );
    }

    const std::size_t expected_size =
        phi.size();

    std::vector<unsigned char> seen(
        expected_size,
        0
    );

    std::size_t count = 0;

    std::string line;

    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream stream(line);

        std::string k_text;
        std::string phi_text;
        std::string u_text;
        std::string v_text;

        if (
            !std::getline(stream, k_text, ',')
            || !std::getline(stream, phi_text, ',')
            || !std::getline(stream, u_text, ',')
            || !std::getline(stream, v_text)
        )
        {
            throw std::runtime_error(
                "Malformed CSV row."
            );
        }

        const std::size_t k =
            std::stoull(k_text);

        if (k >= expected_size)
        {
            throw std::runtime_error(
                "CSV index out of range."
            );
        }

        if (seen[k] != 0)
        {
            throw std::runtime_error(
                "Duplicate CSV index."
            );
        }

        phi[k] = std::stod(phi_text);
        u[k] = std::stod(u_text);
        v[k] = std::stod(v_text);

        seen[k] = 1;
        ++count;
    }

    if (count != expected_size)
    {
        throw std::runtime_error(
            "Input CSV does not contain nx * ny rows."
        );
    }
}


void write_output_csv(
    const std::string& path,
    const std::vector<double>& phi
)
{
    std::ofstream output(path);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot open output file: "
            + path
        );
    }

    output
        << std::setprecision(
            std::numeric_limits<double>::max_digits10
        );

    output << "k,phi_new\n";

    for (
        std::size_t k = 0;
        k < phi.size();
        ++k
    )
    {
        output
            << k
            << ','
            << phi[k]
            << '\n';
    }
}

}  // namespace


int main(
    const int argc,
    char* argv[]
)
{
    try
    {
        if (
            argc != 9
            && argc != 10
        )
        {
            std::cerr
                << "Usage:\n"
                << "  cpu_stencil_cli "
                << "<nx> <ny> "
                << "<dx> <dy> "
                << "<diffusivity> <dt> "
                << "<input.csv> <output.csv> "
                << "[steps]\n";

            return 1;
        }

        const std::size_t nx =
            std::stoull(argv[1]);

        const std::size_t ny =
            std::stoull(argv[2]);

        const double dx =
            std::stod(argv[3]);

        const double dy =
            std::stod(argv[4]);

        const double diffusivity =
            std::stod(argv[5]);

        const double dt =
            std::stod(argv[6]);

        const std::string input_path =
            argv[7];

        const std::string output_path =
            argv[8];

        const std::size_t steps =
            (
                argc == 10
                ? std::stoull(argv[9])
                : 1
            );

        if (steps == 0)
        {
            throw std::invalid_argument(
                "steps must be >= 1."
            );
        }

        const std::size_t size =
            nx * ny;

        std::vector<double> current(
            size,
            0.0
        );

        std::vector<double> next(
            size,
            0.0
        );

        std::vector<double> u(
            size,
            0.0
        );

        std::vector<double> v(
            size,
            0.0
        );

        read_input_csv(
            input_path,
            current,
            u,
            v
        );

        const stencil::Parameters parameters{
            .nx = nx,
            .ny = ny,
            .dx = dx,
            .dy = dy,
            .diffusivity = diffusivity,
            .dt = dt,
        };

        for (
            std::size_t step = 0;
            step < steps;
            ++step
        )
        {
            stencil::advance_scalar(
                current,
                u,
                v,
                next,
                parameters
            );

            current.swap(next);
        }

        write_output_csv(
            output_path,
            current
        );

        return 0;
    }
    catch (
        const std::exception& exception
    )
    {
        std::cerr
            << "ERROR: "
            << exception.what()
            << '\n';

        return 1;
    }
}
