/* referee_command_functions.h
 *==============================================================================
 * Author: Aaiza A. Khan, Emil Åberg
 * Creation date: 2024-11-26
 * Last modified: 2024-11-26 by Emil Åberg
 * Description: Contains functions to handle RefereeCommand enumerator.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef CENTRALISEDAI_SSLINTERFACE_REFEREECOMMANDFUNCTIONS_H_
#define CENTRALISEDAI_SSLINTERFACE_REFEREECOMMANDFUNCTIONS_H_

/* C++ standard library headers */
#include "string"

/* Project .h files */
#include "../common_types.h"
#include "../ssl-interface/generated/ssl_gc_referee_message.pb.h"

namespace centralised_ai
{
namespace ssl_interface
{

/*!
 * @brief Convert from Referee_Command enum to RefereeCommand enum.
 * 
 * Converts Protobuf defined enumerator Referee_Command to RefereeCommand
 * enumerator defined by this project in common_types.h. This is done for
 * the purpose of keeping ssl-interface from having output in datatypes defined
 * by Protobuf generated code. This will keep classes that make use of
 * ssl-interface from having a dependency to the protobuf generated code.
 * 
 * @in command Referee command represented by enumerator defined by Protobuf.
 * 
 * @return Referee command represented by enumerator defined by the project
 * in common_types.h 
 */
enum RefereeCommand ConvertRefereeCommand(enum Referee_Command command);

/*!
 * @brief Converts RefereeCommand enumerator to string.
 * 
 * Converts RefereeCommand enumerator to string. Useful for printing
 * the referee command in debug purposes.
 *
 * @param[in] referee_command Referee command enumerator.
 *
 * @return Referee command string representation.
 */
std::string RefereeCommandToString(enum RefereeCommand referee_command);

} /* namespace ssl_interface */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_SSLINTERFACE_REFEREECOMMANDFUNCTIONS_H_ */