using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WCFReservationServer
{
    class ReservationService: IReservationService
    {
        public string Reserve(Reservation res)
        {
            Console.WriteLine(res.Sender+"\t"+res.Message);
            return "OK";
        }
    }
}
