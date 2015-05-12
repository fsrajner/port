using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WCFClient2.ServiceReference1;

namespace WCFClient2
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
           
            ReservationServiceClient cl = new ReservationServiceClient();
               var result = cl.Reserve(new Reservation()
                {
	                Sender=textBox1.Text,
	                Message=textBox2.Text
                });

                MessageBox.Show(result);

        }
    }
}
