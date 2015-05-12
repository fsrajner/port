using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace WCFReservationServer
{
    class Program
    {
        static void Main(string[] args)
        {
            var host = new ServiceHost(typeof(ReservationService));
            host.Open();
            Console.WriteLine("started");
            Console.ReadKey();
        }
    }
}
