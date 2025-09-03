#include "omp.h"
#include "./Eigen/Eigen"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

void ReadBinAndOutputJacobianVec(const std::string& directory, Eigen::VectorXd& sigma, Eigen::VectorXd& data, Eigen::VectorXd& loss, Eigen::VectorXd& grad) {
    // **************************************************
    int num_ind_meas, num_param, temp;
    double temp_d;
    Eigen::MatrixXi elec;
    Eigen::VectorXd fcalc;
    Eigen::MatrixXd Jacobian;
    Eigen::VectorXd loss_sig2;
    // Open the binary file and read the data
    std::ifstream inputFile(directory+"/R2_forward.bin", std::ios::binary);
    if (!inputFile) {
        std::cout << "Failed to open R2_forward.bin" << std::endl;
        return;
    }
    // Read num_ind_meas and num_param
    inputFile.read(reinterpret_cast<char*>(&temp), sizeof(temp));
    inputFile.read(reinterpret_cast<char*>(&num_ind_meas), sizeof(num_ind_meas));
    inputFile.read(reinterpret_cast<char*>(&num_param), sizeof(num_param));
    // std::cout << num_ind_meas << "    " << num_param << std::endl;
    // Resize vectors
    fcalc.resize(num_ind_meas);
    elec.resize(4, num_ind_meas);
    Jacobian.resize(num_param, num_ind_meas);
    // Read the data into arrays
    for (int j = 0; j < 4; ++j) {
        inputFile.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        inputFile.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        for(int jj = 0 ; jj < num_ind_meas; jj++ ) {
            inputFile.read(reinterpret_cast<char*>(&elec(j,jj)), sizeof(int));
        }
    }
    // read data
    inputFile.read(reinterpret_cast<char*>(&temp_d), sizeof(temp_d));
    for(int jj = 0 ; jj < num_ind_meas; jj++ ) {
        inputFile.read(reinterpret_cast<char*>(&fcalc(jj)), sizeof(double));
    }
    // read Jacobian    
    for (int j = 0; j < num_param; ++j) {
        inputFile.read(reinterpret_cast<char*>(&temp_d), sizeof(temp_d));
        for(int jj = 0 ; jj < num_ind_meas; jj++ ) {
            inputFile.read(reinterpret_cast<char*>(&Jacobian(j,jj)), sizeof(double));
        }
    }
    inputFile.close();

    // delete
    std::string command = "cd " + directory + "&& rm R2_forward.bin";
    int result = system(command.c_str());
    // **************************************************
    // Write the data to R2_forward.dat
    // std::ofstream outputFile1(directory + "/R2_forward.dat");
    // if (!outputFile1) {
    //     std::cerr << "Failed to open R2_forward.dat" << std::endl;
    //     return;
    // }
    // for (int i = 0; i < num_ind_meas; ++i) {
    //     outputFile1 << i + 1 << "\t\t";
    //     for (int j = 0; j < 4; ++j) {
    //         outputFile1 << elec(j,i) <<"\t\t";
    //     }
    //     outputFile1  << fcalc(i) << std::endl;
    // }
    // outputFile1.close();
    // // Write the data to R2_Jacobian.dat
    // std::ofstream outputFile2(directory +"/R2_Jacobian.dat");
    // if (!outputFile2) {
    //     std::cerr << "Failed to open R2_Jacobian.dat" << std::endl;
    //     return;
    // }
    // for (int j = 0; j < num_param; ++j) {
    //     outputFile2 << j + 1 << std::endl;
    //     for (int i = 0; i < num_ind_meas; ++i) {
    //         outputFile2 << Jacobian(j,i) << std::endl;
    //     }
    // }
    // outputFile2.close();
    // std::cout << "all ok" << std::endl;
    // **************************************************
    // caluate d(Resistance)/dlog(sigma) from dlog(Resistance)/dlog(sigma)  
    
    for (int j = 0; j < num_param; ++j) {
        for(int jj = 0 ; jj < num_ind_meas; jj++ ) {
            Jacobian(j,jj) = Jacobian(j,jj) * fcalc(jj);
        }
    }
    loss.resize(num_ind_meas);
    loss_sig2.resize(num_ind_meas);
    for(int i = 0 ; i < num_ind_meas ; i++) {
        // loss(i) = ( log(fcalc(i)) - data(i) ) / sigma(i) ;
        loss(i) = ( fcalc(i) - data(i) ) / sigma(i) ;
        loss_sig2(i) = loss(i) / sigma(i);
        // loss(i) = ( log_res(i) - data(i) );
    }
    grad = Jacobian * loss_sig2 ;


    // loss.resize(num_ind_meas);
    // loss_sig2.resize(num_ind_meas);
    // for(int i = 0 ; i < num_ind_meas ; i++) {
    //     // loss(i) = ( log_res(i) - data(i) ) / sigma(i) ;
    //     loss(i) = ( log(fcalc(i)) - data(i) ) / sigma(i) ;
    //     loss_sig2(i) = loss(i) / sigma(i);
    //     // loss(i) = ( log_res(i) - data(i) );
    // }
    // grad = Jacobian * loss_sig2 ;
}

void RunAndCalc(const std::string& directory, Eigen::VectorXd& sigma, Eigen::VectorXd& data, Eigen::VectorXd& loss, Eigen::VectorXd& grad) {
    // 构造命令
    std::string command = "cd " + directory + "&& sh run.sh";
    // std::cout << command << std::endl;
    // 执行命令
    int result = system(command.c_str());
    // result = system(std::string("ls").c_str());
    // if (result == 0) {
    //     std::cout << "Program " << directory << "is successfully." << std::endl;
    // } else {
    //     std::cerr << "Program " << directory << "is failed." << std::endl;
    // }
    ReadBinAndOutputJacobianVec(directory, sigma, data, loss, grad);
}

int main() {
    // ****************************************************************************************************
    // you should modify n_threads
    int n_threads = 20;
    // ****************************************************************************************************
    
    // Eigen::initParallel();
    // Eigen::setNbThreads(n_threads);
    int n_obs = 0;
    double nd_obs = 0.0;
    omp_set_num_threads(n_threads);
    std::string str1 = "./R2_" ;
    Eigen::VectorXd data, sigma;
    Eigen::VectorXd loss[n_threads];
    Eigen::VectorXd grad[n_threads];
    // Read data
    std::ifstream inputFile("data_output.bin", std::ios::binary);
    inputFile.read(reinterpret_cast<char*>(&nd_obs), sizeof(double));
    n_obs=int(nd_obs);
    data.resize(n_obs);
    for(int i = 0 ; i < n_obs ; i++ ) {
        inputFile.read(reinterpret_cast<char*>(&data(i)), sizeof(double));
    }
    inputFile.close();
    // Read sigma
    std::ifstream inputFile2("data_sigma.bin", std::ios::binary);
    sigma.resize(n_obs);
    for(int i = 0 ; i < n_obs ; i++ ) {
        inputFile2.read(reinterpret_cast<char*>(&sigma(i)), sizeof(double));
        // std::cout << sigma(i) << std::endl;
    }
    inputFile2.close();
    // int thread_avaible = omp_get_max_threads() - omp_get_num_threads() - 5;
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic,1)
    for( int iter_ = 0 ; iter_ < n_threads ; iter_++ ) {
        RunAndCalc(str1+std::to_string(iter_),sigma, data, loss[iter_], grad[iter_]);
        // system( (str1+std::to_string(iter_)+std::string("/R2_J")).c_str());
        // std::cout << (str1+std::to_string(iter_)+std::string("/R2_J.exe")).c_str() << std::endl;
    } 

    // output data
    std::ofstream outputFile("lossgrad.bin", std::ios::binary);
    for(int i = 0 ; i < n_threads ; i++ ) {
        for (int j = 0 ; j < loss[i].size() ; j++ ) {
            outputFile.write(reinterpret_cast<const char *>(&loss[i][j]), sizeof(double));
        }
        for (int j = 0 ; j < grad[i].size() ; j++ ) {
            outputFile.write(reinterpret_cast<const char *>(&grad[i][j]), sizeof(double));
        }
    }
    outputFile.close();
    return 0;
}
