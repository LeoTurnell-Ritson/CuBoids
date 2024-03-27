#ifndef CUDA_BOID_STRUCT_H
#define CUDA_BOID_STRUCT_H

struct Boid 
{
  float position_x;     // X-coordinate of the position
  float position_y;     // Y-coordinate of the position
  float position_z;     // Z-coordinate of the position

  //float velocity_x;     // X-component of the velocity
  //float velocity_y;     // Y-component of the velocity
  //float velocity_z;     // Z-component of the velocity

  //float acceleration_x; // X-component of the acceleration
  //float acceleration_y; // Y-component of the acceleration
  //float acceleration_z; // Z-component of the acceleration

  // int mass;             // Mass of the boid
  // float max_speed;      // Maximum speed of the boid
  // float max_force;      // Maximum steering force applied to the boid
  // float visibility_range;// Maximum visibility of the Boid
  // int species;          // Speciey type for interactions between Boid of different species, like enemies

};

#endif //CUDA_BOID_STRUCT_H