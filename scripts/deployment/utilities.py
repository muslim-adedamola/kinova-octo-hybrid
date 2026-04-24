import argparse

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2


def parseConnectionArguments(argv=None, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--ip", type=str, help="IP address of destination", default="192.168.1.13")
    parser.add_argument("-u", "--username", type=str, help="username to login", default="admin")
    parser.add_argument("-p", "--password", type=str, help="password to login", default="admin")

    return parser.parse_args(argv)


class DeviceConnection:

    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection(args):
        return DeviceConnection(
            args.ip,
            port=DeviceConnection.TCP_PORT,
            credentials=(args.username, args.password),
        )

    @staticmethod
    def createUdpConnection(args):
        return DeviceConnection(
            args.ip,
            port=DeviceConnection.UDP_PORT,
            credentials=(args.username, args.password),
        )

    def __init__(self, ipAddress, port=TCP_PORT, credentials=("", "")):
        self.ipAddress = ipAddress
        self.port = port
        self.credentials = credentials

        self.sessionManager = None

        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ipAddress, self.port)

        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000
            session_info.connection_inactivity_timeout = 2000

            self.sessionManager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ipAddress)
            self.sessionManager.CreateSession(session_info)

        return self.router

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sessionManager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.sessionManager.CloseSession(router_options)

        self.transport.disconnect()