using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.Text;
using System.Threading.Tasks;

namespace WCFServer
{
    [ServiceContract]
    class CalculatorService
    {
        [OperationContract]
        public int Add(int a, int b) { return a + b; }
    }
}
