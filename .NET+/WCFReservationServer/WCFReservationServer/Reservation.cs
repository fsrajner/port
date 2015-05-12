using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;

namespace WCFReservationServer
{
    [DataContract]
    class Reservation
    {
        [DataMember]
        public string Sender { get; set; }
        [DataMember]
        public string Message { get; set; }

    }
}
