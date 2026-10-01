#include "../src/engine_net.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef _WIN32
#include <sys/socket.h>
#include <unistd.h>
#endif

static void check(bool condition, const std::string& name) {
  if (!condition) {
    std::fprintf(stderr, "FAIL %s\n", name.c_str());
    std::exit(1);
  }
  std::printf("PASS %s\n", name.c_str());
}

#ifdef _WIN32
using socket_handle = SOCKET;
static constexpr socket_handle invalid_socket = INVALID_SOCKET;
static void close_socket(socket_handle descriptor) { closesocket(descriptor); }
#else
using socket_handle = int;
static constexpr socket_handle invalid_socket = -1;
static void close_socket(socket_handle descriptor) { close(descriptor); }
#endif

static void test_listener(const std::string& host) {
  sockaddr_in expected{};
  check(engine_net::address(host, 0, &expected), "parse " + host);
  socket_handle listener = socket(AF_INET, SOCK_STREAM, 0);
  check(listener != invalid_socket, "create listener " + host);
  check(bind(listener, reinterpret_cast<sockaddr*>(&expected), sizeof expected) == 0,
        "bind " + host);
  check(listen(listener, 1) == 0, "listen " + host);
  sockaddr_in actual{};
#ifdef _WIN32
  int address_size = sizeof actual;
#else
  socklen_t address_size = sizeof actual;
#endif
  check(getsockname(listener, reinterpret_cast<sockaddr*>(&actual), &address_size) == 0,
        "getsockname " + host);
  check(actual.sin_addr.s_addr == expected.sin_addr.s_addr,
        "listener uses configured host " + host);
  check(actual.sin_port != 0, "listener has a port " + host);
  sockaddr_in target{};
  check(engine_net::address(engine_net::connect_host(host), ntohs(actual.sin_port), &target),
        "parse client target " + host);
  socket_handle client = socket(AF_INET, SOCK_STREAM, 0);
  check(client != invalid_socket, "create client " + host);
  check(connect(client, reinterpret_cast<sockaddr*>(&target), sizeof target) == 0,
        "paired API can connect " + host);
  close_socket(client);
  close_socket(listener);
}

int main() {
#ifdef _WIN32
  WSADATA winsock{};
  check(WSAStartup(MAKEWORD(2, 2), &winsock) == 0, "winsock startup");
#endif
  sockaddr_in addr{};
  check(engine_net::address(engine_net::kDefaultHost, 8730, &addr) &&
            addr.sin_addr.s_addr == htonl(INADDR_LOOPBACK) && ntohs(addr.sin_port) == 8730,
        "default engine address is loopback:8730, not wildcard");
  for (const char* host : {"", "localhost", "::1", "999.0.0.1", "127.0.0",
                                 "127.00.0.1", "127.0.0.1:8730", "127.0.0.1junk"}) {
    check(!engine_net::address(host, 8730, &addr), "reject invalid host '" + std::string(host) + "'");
  }
  check(!engine_net::address(engine_net::kDefaultHost, -1, &addr) &&
            !engine_net::address(engine_net::kDefaultHost, 65536, &addr),
        "reject invalid ports");
  check(engine_net::connect_host("0.0.0.0") == "127.0.0.1" &&
            engine_net::connect_host("127.0.0.2") == "127.0.0.2",
        "paired API maps only wildcard to loopback");
  test_listener(engine_net::kDefaultHost);
  test_listener("127.0.0.2");
  test_listener("0.0.0.0");
#ifdef _WIN32
  WSACleanup();
#endif
  std::puts("RESULT PASS");
  return 0;
}
