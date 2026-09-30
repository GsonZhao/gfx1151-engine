#pragma once

#include <string>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

namespace engine_net {

inline constexpr const char* kDefaultHost = "127.0.0.1";

inline bool address(const std::string& host, int port, sockaddr_in* addr) {
  *addr = {};
  if (port < 0 || port > 65535) return false;
  addr->sin_family = AF_INET;
  addr->sin_port = htons(static_cast<unsigned short>(port));
  return inet_pton(AF_INET, host.c_str(), &addr->sin_addr) == 1;
}

inline std::string connect_host(const std::string& host) {
  return host == "0.0.0.0" ? kDefaultHost : host;
}

}
