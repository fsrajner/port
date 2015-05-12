using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WCFClient.ServiceReference1;

namespace WCFClient
{
    class Program
    {
        static void Main(string[] args)
        {
            CalculatorServiceClient cl = new CalculatorServiceClient();
            Console.WriteLine(cl.Add(2, 0001344561));
            Console.ReadKey();
        }
    }
}
