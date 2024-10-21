#ifndef SSL_AUTOMATED_REFEREE_H
#define SSL_AUTOMATED_REFEREE_H

/* C++ standard library headers */
#include <string>

/* Project .h files */
#include "ssl_vision_client.h"
#include "../common_types.h"

namespace centralized_ai
{
namespace ssl_interface
{

class AutomatedReferee
{
public:
  /* Constructor */
  AutomatedReferee(VisionClient& vision_client, std::string grsim_ip,
    uint16_t grsim_port);

  /* Analyze the game state using VisionClient and generate commands */
  void AnalyzeGameState();

  /* Start the automated referee, reset score, referee command,
   * robot and ball positions */
  void StartGame(enum Team starting_team, enum Team team_on_positive_half,
    double prepare_kickoff_start_time, int64_t stage_time);

  /* Stop the automated referee, outputs will no longer be updated */
  void StopGame();

  /* Print the current command and score */
  void PrintCommand();

  /* Getters for game state data */
  enum RefereeCommand GetRefereeCommand();
  int GetBlueTeamScore();
  int GetYellowTeamScore();
  float GetBallDesignatedPositionX();
  float GetBallDesignatedPositionY();
  enum Team TeamOnPositiveHalf();

private:
  /* automated referee variables */
  VisionClient& vision_client_;
  std::string grsim_ip;
  uint16_t grsim_port;
  float collision_margin = 12;
  double prepare_kickoff_duration;
  double prepare_kickoff_start_time;
  double time_at_game_start;
  bool game_running;
  enum Team last_kicker_team;
  int64_t stage_time_left;
  int64_t stage_time;

  /* Game state data */
  RefereeCommand referee_command;
  int blue_team_score;
  int yellow_team_score;
  float ball_designated_position_x;
  float ball_designated_position_y;
  enum Team team_on_positive_half;

  /* Helper to translate command to string */
  std::string CommandToString(RefereeCommand command);

  /* Detect if the ball is out of field */
  bool IsBallOutOfField(float ball_x, float ball_y);

  /* Check if ball is currently touching a robot */
  void CheckForCollision();

  /* returns the distance between the specified robot and ball */
  float DistanceToBall(int id, enum Team team);

  /* Return distance to ball and specified point */
  float DistanceToBall(float x, float y);

  bool PrepareKickoffTimePassed();
  bool IsBallInBlueGoal(float ball_x, float ball_y);
  bool IsBallInYellowGoal(float ball_x, float ball_y);
  void SetBallDesignatedPosition();
  bool BallSuccessfullyPlaced();
  void UpdateRefereeCommand();
  int64_t GetStageTimeLeft();
  std::string RefereeCommandToString(enum RefereeCommand referee_command); // Added declaration
};

} /* namespace ssl_interface */
} /* namespace centralized_ai */

#endif /* SSL_AUTOMATED_REFEREE_H */
