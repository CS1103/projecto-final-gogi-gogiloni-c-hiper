//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#include <iostream>
#include <array>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <utility>

namespace utec
{
    namespace algebra
    {
        template<typename T, size_t Rank>
        class Tensor
        {
        public:
            std::array<size_t, Rank> shape_;  
            std::vector<T> data;
        public:
            Tensor(const std::array<size_t, Rank>& _shape) : shape_(_shape) {
                size_t totalSize = 1;
                for (size_t i = 0; i < Rank; ++i) {
                    totalSize *= _shape[i];
                }
                data.resize(totalSize);
            }

            template <typename ... Dims > 
            Tensor (Dims ... dims ){
                std::initializer_list<size_t> dims_list = { static_cast<size_t>(dims)... };
                if (sizeof...(Dims) != Rank) {
                    throw std::runtime_error("Number of dimensions do not match with " + std::to_string(Rank));
                }

                std::copy(dims_list.begin(), dims_list.end(), shape_.begin());

                size_t total_size = 1;
                for (size_t i = 0; i < Rank; ++i) {
                    total_size *= shape_[i];
                }
                data.resize(total_size);
            }
            
            void fill(const T& value) {
                std::fill(data.begin(), data.end(), value);
            }
            
            template < typename ... Idxs > 
            T& operator ()(Idxs ... idxs){
                std::array<size_t, Rank> indices = {static_cast<size_t>(idxs)...};

                size_t index = 0;
                size_t dimension = 1;
                for (size_t i = Rank - 1; i < Rank; --i) {
                    index += indices[i] * dimension;
                    dimension *= shape_[i];
                }

                return data[index];
            }
            template < typename ... Idxs >
            const T& operator()(Idxs ... idxs) const{
                std::array<size_t, Rank> indices = {static_cast<size_t>(idxs)...};

                size_t index = 0;
                size_t dimension = 1;
                for (size_t i = Rank - 1; i < Rank; --i) {
                    index += indices[i] * dimension;
                    dimension *= shape_[i];
                }

                return data[index];
            }

            const std::array<size_t,Rank>& shape() const noexcept{
                return shape_;
            }

            template <typename... Dims>
            void reshape(Dims... dims) {
                if (sizeof...(Dims) != Rank) {
                    throw std::runtime_error("Number of dimensions do not match with " + std::to_string(Rank));
                }

                std::initializer_list<size_t> dims_list = { static_cast<size_t>(dims)... };
                std::copy(dims_list.begin(), dims_list.end(), shape_.begin());

                size_t new_total_size = 1;
                for (size_t i = 0; i < Rank; ++i) {
                    new_total_size *= shape_[i];
                }

                if (new_total_size < data.size()) {
                    data.resize(new_total_size);
                } else if (new_total_size > data.size()) {
                    data.resize(new_total_size, T{});
                }
            }
            void reshape(const std::array<size_t, Rank>& newShape) {
                size_t newTotalSize = 1;
                for (size_t i = 0; i < Rank; ++i) {
                    newTotalSize *= newShape[i];
                }
                if (newTotalSize != data.size()) {
                    throw std::runtime_error("total size does not match");
                }
                shape_ = newShape;
            }

            Tensor operator+(const Tensor& t2) const {
                if constexpr (Rank == 2) {
                    size_t A0 = shape_[0], A1 = shape_[1];
                    size_t B0 = t2.shape_[0], B1 = t2.shape_[1];

                    if ((A0 != B0 && A0 != 1 && B0 != 1) || (A1 != B1 && A1 != 1 && B1 != 1)) {
                        throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
                    }
                    size_t R0 = std::max(A0, B0);
                    size_t R1 = std::max(A1, B1);
                    Tensor result(R0, R1);
                    for (size_t i = 0; i < R0; ++i) {
                        for (size_t j = 0; j < R1; ++j) {
                            T valor1 = data[(A0 == 1 ? 0 : i) * A1 + (A1 == 1 ? 0 : j)];
                            T valor2 = t2.data[(B0 == 1 ? 0 : i) * B1 + (B1 == 1 ? 0 : j)];
                            result.data[i * R1 + j] = valor1 + valor2;
                        }
                    }
                    return result;
                }
                else if constexpr (Rank == 3) {
                    size_t A0 = shape_[0], A1 = shape_[1], A2 = shape_[2];
                    size_t B0 = t2.shape_[0], B1 = t2.shape_[1], B2 = t2.shape_[2];
                    if ((A0 != B0 && A0 != 1 && B0 != 1) || (A1 != B1 && A1 != 1 && B1 != 1) || (A2 != B2 && A2 != 1 && B2 != 1)) {
                        throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
                    }
                    size_t R0 = std::max(A0, B0);
                    size_t R1 = std::max(A1, B1);
                    size_t R2 = std::max(A2, B2);
                    Tensor result(R0, R1, R2);
                    for (size_t i = 0; i < R0; ++i) {
                        for (size_t j = 0; j < R1; ++j) {
                            for (size_t k = 0; k < R2; ++k) {
                                T valor1 = data[((A0 == 1 ? 0 : i) * A1 + (A1 == 1 ? 0 : j)) * A2 + (A2 == 1 ? 0 : k)];
                                T valor2 = t2.data[((B0 == 1 ? 0 : i) * B1 + (B1 == 1 ? 0 : j)) * B2 + (B2 == 1 ? 0 : k)];
                                result.data[(i * R1 + j) * R2 + k] = valor1 + valor2;
                            }
                        }
                    }
                    return result;
                }
                else {
                    throw std::runtime_error("Broadcast not implemented for Rank > 3");
                }
            }
            
            Tensor operator-(const Tensor& t2) const {
                if constexpr (Rank == 2) {
                    size_t A0 = shape_[0], A1 = shape_[1];
                    size_t B0 = t2.shape_[0], B1 = t2.shape_[1];

                    if ((A0 != B0 && A0 != 1 && B0 != 1) || (A1 != B1 && A1 != 1 && B1 != 1)) {
                        throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
                    }
                    size_t R0 = std::max(A0, B0);
                    size_t R1 = std::max(A1, B1);
                    Tensor result(R0, R1);
                    for (size_t i = 0; i < R0; ++i) {
                        for (size_t j = 0; j < R1; ++j) {
                            T valor1 = data[(A0 == 1 ? 0 : i) * A1 + (A1 == 1 ? 0 : j)];
                            T valor2 = t2.data[(B0 == 1 ? 0 : i) * B1 + (B1 == 1 ? 0 : j)];
                            result.data[i * R1 + j] = valor1 - valor2;
                        }
                    }
                    return result;
                }
                else if constexpr (Rank == 3) {
                    size_t A0 = shape_[0], A1 = shape_[1], A2 = shape_[2];
                    size_t B0 = t2.shape_[0], B1 = t2.shape_[1], B2 = t2.shape_[2];
                    if ((A0 != B0 && A0 != 1 && B0 != 1) || (A1 != B1 && A1 != 1 && B1 != 1) || (A2 != B2 && A2 != 1 && B2 != 1)) {
                        throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
                    }
                    size_t R0 = std::max(A0, B0);
                    size_t R1 = std::max(A1, B1);
                    size_t R2 = std::max(A2, B2);
                    Tensor result(R0, R1, R2);
                    for (size_t i = 0; i < R0; ++i) {
                        for (size_t j = 0; j < R1; ++j) {
                            for (size_t k = 0; k < R2; ++k) {
                                T valor1 = data[((A0 == 1 ? 0 : i) * A1 + (A1 == 1 ? 0 : j)) * A2 + (A2 == 1 ? 0 : k)];
                                T valor2 = t2.data[((B0 == 1 ? 0 : i) * B1 + (B1 == 1 ? 0 : j)) * B2 + (B2 == 1 ? 0 : k)];
                                result.data[(i * R1 + j) * R2 + k] = valor1 - valor2;
                            }
                        }
                    }
                    return result;
                }
                else {
                    throw std::runtime_error("Broadcast not implemented for Rank > 3");
                }
            }


            Tensor operator*(const Tensor& t2) const {
                if constexpr (Rank == 2) {
                    size_t A0 = shape_[0], A1 = shape_[1];
                    size_t B0 = t2.shape_[0], B1 = t2.shape_[1];

                    if ((A0 != B0 && A0 != 1 && B0 != 1) || (A1 != B1 && A1 != 1 && B1 != 1)) {
                        throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
                    }

                    size_t R0 = std::max(A0, B0);
                    size_t R1 = std::max(A1, B1);
                    Tensor result(R0, R1);

                    for (size_t i = 0; i < R0; ++i) {
                        for (size_t j = 0; j < R1; ++j) {
                            T valor1 = data[(A0 == 1 ? 0 : i) * A1 + (A1 == 1 ? 0 : j)];
                            T valor2 = t2.data[(B0 == 1 ? 0 : i) * B1 + (B1 == 1 ? 0 : j)];
                            result.data[i * R1 + j] = valor1 * valor2;
                        }
                    }

                    return result;
                }
                else if constexpr (Rank == 3) {
                    size_t A0 = shape_[0], A1 = shape_[1], A2 = shape_[2];
                    size_t B0 = t2.shape_[0], B1 = t2.shape_[1], B2 = t2.shape_[2];

                    if ((A0 != B0 && A0 != 1 && B0 != 1) || (A1 != B1 && A1 != 1 && B1 != 1) || (A2 != B2 && A2 != 1 && B2 != 1)) {
                        throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
                    }

                    size_t R0 = std::max(A0, B0);
                    size_t R1 = std::max(A1, B1);
                    size_t R2 = std::max(A2, B2);
                    Tensor result(R0, R1, R2);

                    for (size_t i = 0; i < R0; ++i) {
                        for (size_t j = 0; j < R1; ++j) {
                            for (size_t k = 0; k < R2; ++k) {
                                T valor1 = data[((A0 == 1 ? 0 : i) * A1 + (A1 == 1 ? 0 : j)) * A2 + (A2 == 1 ? 0 : k)];
                                T valor2 = t2.data[((B0 == 1 ? 0 : i) * B1 + (B1 == 1 ? 0 : j)) * B2 + (B2 == 1 ? 0 : k)];
                                result.data[(i * R1 + j) * R2 + k] = valor1 * valor2;
                            }
                        }
                    }

                    return result;
                }
                else {
                    throw std::runtime_error("Broadcast not implemented for Rank > 3");
                }
            }  

            Tensor operator+(T escalar) const {
                Tensor result(shape_);
                for (size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] + escalar;
                }
                return result;
            }

            Tensor operator-(T escalar) const {
                Tensor result(shape_);
                for (size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] - escalar;
                }
                return result;
            }

            Tensor operator*(T escalar) const {
                Tensor result(shape_);
                for (size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] * escalar;
                }
                return result;
            }

            Tensor operator/(T escalar) const {
                Tensor result(shape_);
                for (size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] / escalar;
                }
                return result;
            }

            friend Tensor operator+(T escalar, const Tensor& t) {
                return t + escalar;
            }
            Tensor& operator=(std::initializer_list<T> list) {
                if (list.size() != data.size()) {
                    throw std::runtime_error("Data size does not match tensor size");
                }
                std::copy(list.begin(), list.end(), data.begin());
                return *this;
            }

            friend std::ostream& operator<<(std::ostream& os, const Tensor<T, Rank>& t) {
                if constexpr (Rank == 1) {
                    for (size_t i = 0; i < t.shape_[0]; ++i) {
                        os << t.data[i] << (i + 1 < t.shape_[0] ? " " : "");
                    }
                    return os;
                } else if constexpr (Rank == 2) {
                    os << "{\n";
                    size_t R = t.shape_[0], C = t.shape_[1];
                    for (size_t i = 0; i < R; ++i) {
                        for (size_t j = 0; j < C; ++j) {
                            os << t.data[i * C + j];
                            if (j + 1 < C) os << " ";
                        }
                        os << "\n";
                    }
                    os << "}";
                    return os;
                } else if constexpr (Rank == 3) {
                    os << "{\n";
                    size_t D0 = t.shape_[0], D1 = t.shape_[1], D2 = t.shape_[2];
                    for (size_t i = 0; i < D0; ++i) {
                        os << "{\n";
                        for (size_t r = 0; r < D1; ++r) {
                            for (size_t c = 0; c < D2; ++c) {
                                os << t.data[(i * D1 + r) * D2 + c];
                                if (c + 1 < D2) os << " ";
                            }
                            os << "\n";
                        }
                        os << "}\n";
                    }
                    os << "}";
                    return os;
                } else if constexpr (Rank == 4) {
                    os << "{\n";
                    size_t D0 = t.shape_[0], D1 = t.shape_[1], D2 = t.shape_[2], D3 = t.shape_[3];
                    for (size_t i0 = 0; i0 < D0; ++i0) {
                        os << "{\n";
                        for (size_t i1 = 0; i1 < D1; ++i1) {
                            os << "{\n";
                            for (size_t i2 = 0; i2 < D2; ++i2) {
                                for (size_t i3 = 0; i3 < D3; ++i3) {
                                    size_t flat = ((i0 * D1 + i1) * D2 + i2) * D3 + i3;
                                    os << t.data[flat] << (i3 + 1 < D3 ? " " : "");
                                }
                                os << "\n";
                            }
                            os << "}\n";
                        }
                        os << "}\n";
                    }
                    os << "}";
                    return os;
                } else {
                    os << "{";
                    for (size_t i = 0; i < t.data.size(); ++i) {
                        os << t.data[i];
                        if (i + 1 < t.data.size()) os << ", ";
                    }
                    os << "}";
                    return os;
                }
            }

            size_t size() const {
                size_t total = 1;
                for (size_t d : shape_){
                    total *= d;
                }

                return total;
            }

            auto begin(){
                return data.begin();  
            }
            auto end(){
                return data.end();   
             }
            auto cbegin() const{ 
                return data.cbegin(); 
            }
            auto cend() const{ 
                return data.cend();  
            }
            ~Tensor() {}
        };


        template<typename T, size_t Rank>
        Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& tensor) {
            if (Rank < 2) {
                throw std::runtime_error("Cannot transpose 1D tensor: need at least 2 dimensions");
            }

            std::array<size_t, Rank> original_shape = tensor.shape();
            std::array<size_t, Rank> transposed_shape = original_shape;
            std::swap(transposed_shape[Rank - 2], transposed_shape[Rank - 1]);

            Tensor<T, Rank> result(transposed_shape);

            std::array<size_t, Rank> original_strides{}, transposed_strides{};
            original_strides[Rank - 1] = transposed_strides[Rank - 1] = 1;

            for (int i = static_cast<int>(Rank) - 2; i >= 0; --i) {
                original_strides[i] = original_strides[i + 1] * original_shape[i + 1];
                transposed_strides[i] = transposed_strides[i + 1] * transposed_shape[i + 1];
            }

            size_t total_elements = result.size();
            for (size_t flat_index = 0; flat_index < total_elements; ++flat_index) {
                size_t remaining = flat_index;
                std::array<size_t, Rank> transposed_coords{};

                for (size_t k = 0; k < Rank; ++k) {
                    transposed_coords[k] = remaining / transposed_strides[k];
                    remaining %= transposed_strides[k];
                }

                std::array<size_t, Rank> original_coords = transposed_coords;
                std::swap(original_coords[Rank - 2], original_coords[Rank - 1]);

                size_t original_index = 0;
                for (size_t k = 0; k < Rank; ++k) {
                    original_index += original_coords[k] * original_strides[k];
                }

                result.begin()[flat_index] = tensor.cbegin()[original_index];
            }

            return result;
        }
        template<typename T, size_t Rank>
        Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& a, const Tensor<T, Rank>& b) {

            auto shape_a = a.shape();
            auto shape_b = b.shape();

            size_t rows_a = shape_a[Rank - 2];
            size_t common_dim = shape_a[Rank - 1];
            size_t inner_b = shape_b[Rank - 2];
            size_t cols_b = shape_b[Rank - 1];

            if (common_dim != inner_b) {
                throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
            }

            std::array<size_t, Rank> result_shape = shape_a;
            for (size_t i = 0; i + 2 < Rank; ++i) {
                if (shape_a[i] != shape_b[i]) {
                    throw std::runtime_error("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
                }
                result_shape[i] = shape_a[i]; 
            }
            result_shape[Rank - 2] = rows_a;
            result_shape[Rank - 1] = cols_b;

            Tensor<T, Rank> result(result_shape);

            std::array<size_t, Rank> stride_a{}, stride_b{}, stride_r{};
            stride_a[Rank - 1] = stride_b[Rank - 1] = stride_r[Rank - 1] = 1;

            for (int i = static_cast<int>(Rank) - 2; i >= 0; --i) {
                stride_a[i] = stride_a[i + 1] * shape_a[i + 1];
                stride_b[i] = stride_b[i + 1] * shape_b[i + 1];
                stride_r[i] = stride_r[i + 1] * result_shape[i + 1];
            }

            size_t batch_count = 1;
            for (size_t i = 0; i + 2 < Rank; ++i) {
                batch_count *= result_shape[i];
            }

            for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
                std::array<size_t, Rank> batch_coords{};
                size_t temp = batch_index;

                for (size_t i = 0; i + 2 < Rank; ++i) {
                    batch_coords[i] = temp % result_shape[i];
                    temp /= result_shape[i];
                }

                size_t offset_a = 0, offset_b = 0, offset_r = 0;
                for (size_t i = 0; i + 2 < Rank; ++i) {
                    size_t idx_a = (shape_a[i] == 1) ? 0 : batch_coords[i];
                    size_t idx_b = (shape_b[i] == 1) ? 0 : batch_coords[i];
                    offset_a += idx_a * stride_a[i];
                    offset_b += idx_b * stride_b[i];
                    offset_r += batch_coords[i] * stride_r[i];
                }

                for (size_t i = 0; i < rows_a; ++i) {
                    for (size_t j = 0; j < cols_b; ++j) {
                        T total = T();
                        for (size_t k = 0; k < common_dim; ++k) {
                            size_t idx_a = offset_a + i * stride_a[Rank - 2] + k * stride_a[Rank - 1];
                            size_t idx_b = offset_b + k * stride_b[Rank - 2] + j * stride_b[Rank - 1];
                            total += a.cbegin()[idx_a] * b.cbegin()[idx_b];
                        }
                        size_t idx_r = offset_r + i * stride_r[Rank - 2] + j * stride_r[Rank - 1];
                        result.begin()[idx_r] = total;
                    }
                }
            }

            return result;
        }

    } // namespace algebra
    
} // namespace utec



#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
