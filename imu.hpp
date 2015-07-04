#include <matrix.hpp> 
#include <quat.hpp>

template <class T>
struct params {
    T var_update_orient;
    T var_update_angvel;
    T var_meas_orient;
    T var_meas_angvel;
};

template <class T>
void gn(T earth_grav[3], T earth_mag[3], T body_grav[3], T body_mag[3], T guess_q[4]);

template <class T>
void kalman(struct params<T> params, T state[7], T covariance[7][7], T meas[7], T dt);

template <class T>
void multv_position(size_t rows, size_t cols, T *m, T *v, size_t cols_out, T *out, size_t col_offset, size_t row_offset){
    size_t i, j;
    for(i=0; i<rows; i++){
        T result = get_zero<T>();
        for(j=0; j<cols; j++){
            result += *(m + i*cols + j) * v[j];
        }
        *(out + (row_offset + i)*cols_out + col_offset) = result;
    }
}

//Assumes all measurements are normalized
template <class T>
void gn(T earth_grav[3], T earth_mag[3], T body_grav[3], T body_mag[3], T guess_q[4]){

    fix16Exc grav_weight = 20;
    fix16Exc mag_weight  = 1;

    size_t i, j, k;
    for(i=0; i<50; i++){
        normalize(4, guess_q, guess_q);

        T body_grav_r[3], body_mag_r[3];
        rotate(guess_q, body_grav, body_grav_r);
        rotate(guess_q, body_mag,  body_mag_r);

        T resid[6];
        subv(3, earth_grav, body_grav_r, resid);
        subv(3, earth_mag,  body_mag_r,  resid+3);

        scalev(grav_weight, 3, resid, resid);
        scalev(mag_weight, 3, resid+3, resid+3);

        T ja[3][3] = {{ guess_q[0], -guess_q[3], guess_q[2]}, {guess_q[3],  guess_q[0], -guess_q[1]}, {-guess_q[2], guess_q[1],  guess_q[0]}};
        T jb[3][3] = {{ guess_q[1],  guess_q[2], guess_q[3]}, {guess_q[2], -guess_q[1], -guess_q[0]}, { guess_q[3], guess_q[0], -guess_q[1]}};
        T jc[3][3] = {{-guess_q[2],  guess_q[1], guess_q[0]}, {guess_q[1],  guess_q[2],  guess_q[3]}, {-guess_q[0], guess_q[3], -guess_q[2]}};
        T jd[3][3] = {{-guess_q[3], -guess_q[0], guess_q[1]}, {guess_q[0], -guess_q[3],  guess_q[2]}, { guess_q[1], guess_q[2],  guess_q[3]}};
        scale<T>(2, 3, 3, (T *)ja, (T *)ja);
        scale<T>(2, 3, 3, (T *)jb, (T *)jb);
        scale<T>(2, 3, 3, (T *)jc, (T *)jc);
        scale<T>(2, 3, 3, (T *)jd, (T *)jd);

        T jacobian[6][4];

        multv_position(3, 3, (T *)ja, body_grav, 4, (T *)jacobian, 0, 0);
        multv_position(3, 3, (T *)jb, body_grav, 4, (T *)jacobian, 1, 0);
        multv_position(3, 3, (T *)jc, body_grav, 4, (T *)jacobian, 2, 0);
        multv_position(3, 3, (T *)jd, body_grav, 4, (T *)jacobian, 3, 0);

        multv_position(3, 3, (T *)ja, body_mag,  4, (T *)jacobian, 0, 3);
        multv_position(3, 3, (T *)jb, body_mag,  4, (T *)jacobian, 1, 3);
        multv_position(3, 3, (T *)jc, body_mag,  4, (T *)jacobian, 2, 3);
        multv_position(3, 3, (T *)jd, body_mag,  4, (T *)jacobian, 3, 3);

        for(j=0; j<3; j++){
            for(k=0; k<4; k++){
                jacobian[j][k] = jacobian[j][k] * grav_weight;
            }
        }

        for(; j<6; j++){
            for(k=0; k<4; k++){
                jacobian[j][k] = jacobian[j][k] * mag_weight;
            }
        }

        scale<T>(-1, 6, 4, (T *)jacobian, (T *)jacobian);

        T r[4][4];
        qr(6, 4, (T *)jacobian, (T *)r);

        T result[4];
        solve(6, 4, (T *)jacobian, (T *)r, resid, result);
        subv(4, guess_q, result, guess_q);

        if(result[0] < T(0.01) && result[1] < T(0.01) && result[2] < T(0.01) && result[3] < T(0.01)){
            //Serial.printf("Iters: %d\r\n", i);
            break;
        }
    }

    normalize(4, guess_q, guess_q);
}

template <class T>
void computeJacobian(T dt, T state[7], T jacobian[7][7]){

    T w=state[0], x=state[1], y=state[2], z=state[3], p=state[4], q=state[5], r=state[6];
    T jacobian2[7][7] = {
        {0, -p, -q, -r, -x, -y, -z},
        {p,  0,  r, -q,  w, -z,  y},
        {q, -r,  0,  p,  z,  w, -x},
        {r,  q, -p,  0, -y,  x,  w},
        {0,  0,  0,  0,  0,  0,  0},
        {0,  0,  0,  0,  0,  0,  0},
        {0,  0,  0,  0,  0,  0,  0}
    };

    int i, j;
    for(i=0; i<7; i++){
        for(j=0; j<7; j++){
            jacobian[i][j] = jacobian2[i][j];
        }
    }

    scale<T>(0.5 * dt, 7, 7, (T *)jacobian, (T *)jacobian);

    for(i=0; i<7; i++){
        jacobian[i][i] = 1;
    }
}

template <class T>
void computeL(T dt, T state[7], T l[7][3]){
    T w=state[0], x=state[1], y=state[2], z=state[3];
    T l2[7][3] = {
        {-x, -y, -z},
        { w, -z,  y},
        { z,  w, -x},
        {-y,  x,  w},
        { 1,  0,  0},
        { 0,  1,  0},
        { 0,  0,  1}
    };

    int i, j;
    for(i=0; i<7; i++){
        for(j=0; j<3; j++){
            l[i][j] = l2[i][j];
        }
    }

    for(i=0; i<4; i++){
        for(j=0; j<3; j++){
            l[i][j] = l[i][j] * 0.25 * dt * dt;
        }
    }

    for(i=4; i<7; i++){
        for(j=0; j<3; j++){
            l[i][j] = l[i][j] * dt;
        }
    }
}

template <class T>
void updateState(T dt, T state[7]){
    T rot_quat[4];
    axis_to_quat(dt, state + 4, rot_quat);
    T temp[4];
    mul_q(state, rot_quat, temp);
    normalize(4, temp, state);
}

template <class T>
struct kalman_state{
    T state[7];
    T covariance[7][7];
};

template <class T>
void init_kalman_state(T var_pos_init, T var_angvel_init, struct kalman_state<T> *kalman_state){
    int i, j;

    kalman_state->state[0] = 1;
    for(i=1; i<7; i++)
        kalman_state->state[i] = 0;

    for(i=0; i<7; i++){
        for(j=0; j<7; j++){
            kalman_state->covariance[i][j] = 0;
        }
    }

    for(i=0; i<4; i++)
        kalman_state->covariance[i][i] = var_pos_init;

    for(; i<7; i++)
        kalman_state->covariance[i][i] = var_angvel_init;
}

template <class T>
void kalman(struct params<T> *params, T state[7], T covariance[7][7], T meas[7], T dt){
    // Compute jacobians
    T jacobian[7][7];
    computeJacobian(dt, state, jacobian);

    T l[7][3];
    computeL(dt, state, l);

    // Update state
    updateState(dt, state);

    // Update covariance
    T temp[7][7];
    mmult_yt(7, 7, 7, (T *)covariance, (T *)jacobian, (T *)temp);
    mmult(7, 7, 7, (T *)jacobian, (T *)temp, (T *)covariance);
    T var_orient = params->var_update_orient * dt * dt;
    T var_angvel = params->var_update_angvel * dt * dt;
    T q[7] = {
        var_orient, var_orient, var_orient, var_orient, 
        var_angvel, var_angvel, var_angvel
    };
    size_t i, j;
    for(i=0; i<7; i++){
        covariance[i][i] += q[i];
    }

    // State residual
    T meas_resid[7];
    subv(7, meas, state, meas_resid);

    // Covariance residual
    T cov_resid[7][7];
    T r[7] = {
        params->var_meas_orient, params->var_meas_orient, params->var_meas_orient, params->var_meas_orient, 
        params->var_meas_angvel, params->var_meas_angvel, params->var_meas_angvel
    };
    for(i=0; i<7; i++){
        for(j=0; j<7; j++){
            cov_resid[i][j] = covariance[i][j];
        }
    }

    for(i=0; i<7; i++){
        cov_resid[i][i] += r[i];
    }

    // Compute kalman gain
    T kalman_gain[7][7];
    T rr[7][7];
    qr(7, 7, (T *)cov_resid, (T *)rr);
    solve(7, 7, 7, (T *)cov_resid, (T *)rr, (T *)covariance, (T *)kalman_gain);
    transpose_inplace(7, 7, (T *)kalman_gain);

    // Correct next state
    T temp2[7];
    multv(7, 7, (T *)kalman_gain, meas_resid, temp2);
    addv(7, state, temp2, state);

    // Correct next covariance
    scale<T>(-1, 7, 7, (T *)kalman_gain, (T *)kalman_gain);
    for(i=0; i<7; i++){
        kalman_gain[i][i] += 1;
    }

    T tmp_cov[7][7];
    mmult(7, 7, 7, (T *)kalman_gain, (T *)covariance, (T *)tmp_cov);
    for(i=0; i<7; i++){
        for(j=0; j<7; j++){
            covariance[i][j] = tmp_cov[i][j];
        }
    }
}

template <class T>
struct calibration{
    T earth_grav[3];
    T earth_mag[3];
    T stat_gyro[3];
};

template <class T>
struct measurements{
    T body_grav[3];
    T body_mag[3];
    T body_gyro[3];
};

template <class T>
void imu(struct calibration<T> *calibration, struct params<T> *params, struct measurements<T> *measurements, struct kalman_state<T> *kalman_state, T dt){
    T meas[7];
    int i;
    for(i=0; i<4; i++){
        meas[i] = kalman_state->state[i];
    }

    jmp_buf jb;
    jmp_buf *last_jb = overflow_exc;
    overflow_exc = &jb;
    int exc;

    if(!(exc = setjmp(jb))){
        gn(calibration->earth_grav, calibration->earth_mag, measurements->body_grav, measurements->body_mag, meas);
    } else {
        Serial.printf("Gauss Newtop exception: %d\n", exc);
        overflow_exc = last_jb;
        longjmp(*overflow_exc, exc);
    }

    for(i=0; i<3; i++){
        meas[i+4] = measurements->body_gyro[i];
    }

    if(!(exc = setjmp(jb))){
        kalman(params, kalman_state->state, kalman_state->covariance, meas, dt);
    } else {
        Serial.printf("Kalman filter exception: %d\n", exc);
        overflow_exc = last_jb;
        longjmp(*overflow_exc, exc);
    }

    overflow_exc = last_jb;
}
