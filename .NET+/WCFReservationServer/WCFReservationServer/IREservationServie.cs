using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;

namespace WCFReservationServer
{
    [ServiceContract]
    interface IReservationService
    {
        [OperationContract]
        string Reserve(Reservation res);
    }
}
