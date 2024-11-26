/* simulation_interface_test.cc
 *==============================================================================
 * Author: Aaiza Aziz Khan and Shruthi Puthiya Kunnon
 * Creation date: 2024-11-01
 * Last modified: 2024-11-01 by Aaiza Aziz Khan
 * Description: Test suite for simulation interface
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "../../src/simulation-interface/simulation_interface.h"

/* Other .h files */
#include "gtest/gtest.h"

    class TestableSimulationInterface 
        : public centralised_ai::simulation_interface::SimulationInterface {
      public:
        TestableSimulationInterface(std::string ip, uint16_t port, int id, 
            centralised_ai::Team team)
                : SimulationInterface(ip, port, id, team) {}
          GrSimPacket CallCreateProtoPacket() {
          return CreateProtoPacket();
          }

    };

/* Test to verify a packet is constructed correctly */
TEST(SimulationInterfaceTest, CreateProtoPacketTest) {
  TestableSimulationInterface sim_interface("127.0.0.1", 10001, 1, 
        centralised_ai::Team::kYellow);

  /* Set values using the provided setter methods */
  sim_interface.SetKickerSpeed(5.0f);
  sim_interface.SetSpinnerOn(true);
  sim_interface.SetVelocity(2.0f, 3.0f, 1.0f);

  /* Call CreateProtoPacket */
  GrSimPacket packet = sim_interface.CallCreateProtoPacket();

  /* Check the values in the packet */
  EXPECT_EQ(
      packet.commands().is_team_yellow(), true);
  EXPECT_FLOAT_EQ(
      packet.commands().robot_commands(0).kick_speed_x(), 5.0f);
  EXPECT_FLOAT_EQ(
      packet.commands().robot_commands(0).kick_speed_z(), 0.0f);
  EXPECT_EQ(
      packet.commands().robot_commands(0).spinner(), true);
  EXPECT_FLOAT_EQ(
      packet.commands().robot_commands(0).vel_tangent(), 2.0f);
  EXPECT_FLOAT_EQ(
      packet.commands().robot_commands(0).vel_normal(), 3.0f);
  EXPECT_FLOAT_EQ(
      packet.commands().robot_commands(0).vel_angular(), 1.0f);
}
